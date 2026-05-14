# `generate-qa.py` — Question / Answer / Citation generation from logical-chunk artifacts

Standalone script that turns the per-chunk artifacts emitted by `extract_artifacts.py` (Step 2 of the mode-3 logical-chunk flow) into a corpus of `(question, answer, citation)` triples suitable for fine-tuning embedding models.

For the design rationale, see `plans/plan-generate-qa.md`. For the formal contract (FRs / NFRs / ACs), see `plans/srs-generate-qa.md`.

---

## What it does

Two phases, both async with bounded concurrency:

1. **Phase 1 — Q&A generation.** For each `(context, artifact)` pair, send one LLM call and get back 3-5 questions with answers. Saves to `<output_dir>/generated-questions_qa_only.json`.
2. **Phase 2 — Citation extraction.** For each generated question, send one LLM call to extract a verbatim citation from the surrounding context. Saves to `<output_dir>/generated-questions_with_citations.json`.

After both phases complete, the script sorts by `(doc_id, u_ctx_id, artifact_id)` with a natural-integer key, renumbers questions per ctx (`{u_ctx_id}-q-{n}`), and writes the final files:
- `generated-questions.json` — full schema, sorted, renumbered.
- `generated-questions_wo_context.json` — same as above without the `context_text` field (smaller; for downstream consumers that don't need it).
- `generated-questions.csv` — flattened columns for spreadsheets / pandas.

---

## Pipeline at a glance

```
*-logic-ctx.json + *-logic-artifacts.json
        ↓
build_tasks  →  one (context, artifact) task per item, plus optional synthetic "summary" task
        ↓
Phase 1 (asyncio.Semaphore(max_concurrent_qa))
   per task: nemo_qa-gen-artifact.txt → 3-5 questions
        ↓
generated-questions_qa_only.json    (intermediate, TEMP- IDs)
        ↓
Phase 2 (asyncio.Semaphore(max_concurrent_citations))
   per question: nemo_extract-citation.txt → verbatim citation
        ↓
generated-questions_with_citations.json   (intermediate, TEMP- IDs)
        ↓
sort by (doc_id, u_ctx_id, artifact_id) using natural-integer key
renumber question_id as {u_ctx_id}-q-{n}
        ↓
generated-questions.json + generated-questions_wo_context.json + generated-questions.csv
```

---

## Prerequisites

### Python and packages

Python 3.11+ (for `tomllib`). Required packages on Linux / macOS:

```bash
pip install loguru==0.7.3 pandas==2.3.0 python-dotenv==1.1.0 \
            openai==1.91.0 pydantic==2.11.7 ollama==0.5.1 tqdm
```

Optional, only if a `gemini-*` model is selected:

```bash
pip install google-generativeai
```

(This is the **legacy** SDK, *not* `google-genai` — they are different packages.)

### Environment variables (in `.env` or the shell)

- `OPENAI_API_KEY` — required when `model_qa` or `model_citations` starts with `gpt`.
- `GOOGLE_API_KEY` (or `GOOGLE_API_KEY_V13` as fallback) — required when a `gemini-*` model is selected.
- No env var required for Ollama; the daemon must be reachable at `http://{host}:{port}` (default `localhost:11434`).

### Input files

For each document the script reads two sibling files from `chunk_dir`:

```
<chunk_dir>/
  <doc_id>-logic-ctx.json         ← Step 1 output (_nemo.py --sdg-logical)
  <doc_id>-logic-artifacts.json   ← Step 2 output (extract_artifacts.py)
```

Pairs missing the artifacts file are skipped with a warning.

---

## Quickstart

```bash
# 1. Install dependencies (see above)

# 2. Make sure Steps 1 and 2 have run:
.venv/bin/python _nemo.py --sdg-logical --cfg cfg/nemo.toml
.venv/bin/python extract_artifacts.py

# 3. Configure generate-qa.toml (chunk_dir, output_dir, models, etc.)

# 4. Run:
.venv/bin/python generate-qa.py --config generate-qa.toml
```

---

## CLI

```
python generate-qa.py [--config PATH] [--host STR] [--port INT]
                      [--log-level {DEBUG,INFO,WARNING,ERROR}]
```

| Flag | Default | Purpose |
|---|---|---|
| `--config` | `generate-qa.toml` | TOML config path. |
| `--host` | `localhost` | Ollama daemon host. |
| `--port` | `11434` | Ollama daemon port. |
| `--log-level` | `INFO` | Console log verbosity. File log (`logs/generate-qa.log`) stays at DEBUG regardless. Use `WARNING` to see just the progress bar; `DEBUG` for everything. |

Force regeneration: delete `generated-questions_qa_only.json` and/or `generated-questions_with_citations.json` from `output_dir`. There is no `--overwrite` flag.

---

## Configuration (`generate-qa.toml`)

```toml
[generate-qa]
# Phase control — set to false to skip a phase. Useful when iterating on prompts.
generate_qa = true
extract_citations = true

# Inputs: directory containing *-logic-ctx.json + *-logic-artifacts.json pairs.
chunk_dir = "./data/nemo_briefs_20260429/doc-chunks_256_random_logical"

# Output directory; will be created if missing.
output_dir = "./data/nemo_briefs_20260429/qa-gen"
output_file = "generated-questions.json"
output_qa_file = "generated-questions_qa_only.json"
output_csv_file = "generated-questions.csv"

# Prompts (relative to repo root).
question_generate_prompt = "./prompts/nemo_qa-gen-artifact.txt"
extract_citation_prompt  = "./prompts/nemo_extract-citation.txt"

# Models — see "Provider routing" below.
model_qa = "gpt-4o-mini"
model_citations = "gpt-4o-mini"

# Concurrency / periodic save.
max_concurrent_qa = 20                 # Phase 1 in-flight cap
max_concurrent_citations = 20          # Phase 2 in-flight cap
periodic_save_interval_qa = 25         # save intermediate every N completed tasks
periodic_save_interval_citations = 100 # save intermediate every N completed questions

# Artifact selection. Any langextract category from -logic-artifacts.json is valid;
# anything not listed here is ignored. See docs/qa-generation.md for the full 21-class
# taxonomy.
artifact_categories = [
  "finding", "issue", "method", "procedure",
  "recommendation", "best_practice", "rationale",
  "requirement", "definition", "constraint", "parameter", "condition",
]

# Add one synthetic task per context built from chunk_signals.summary.
include_summary_element = true

# Hard cap on artifacts per context (after category filter, before summary). 0 = no cap.
max_artifacts_per_ctx = 0

# Optional symbol normalization applied to context text before sending to the LLM.
replace_symbols = false
[[generate-qa.symbols]]
values       = ["±", "≥", "≤", "µm"]
replace_with = ["+/-", ">=", "<=", "um"]
```

### Periodic save semantics

| Field | Counts | Default |
|---|---|---|
| `periodic_save_interval_qa` | Completed **(context, artifact) tasks** in Phase 1. One task ≈ 3-5 questions. | 25 |
| `periodic_save_interval_citations` | Completed **questions** in Phase 2. One question = one citation call. | 100 |

The final save at end-of-phase happens unconditionally. Crashes between the last periodic save and the end-of-phase save lose up to N tasks/questions of work.

---

## Provider routing (model name → provider)

The script auto-routes based on the model name string:

| Model name pattern | Provider | Examples |
|---|---|---|
| Starts with `gpt`, **no colon** | OpenAI Responses API | `gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo` |
| Starts with `gemini`, **no colon** | Google Gemini (legacy SDK) | `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-1.5-pro-002` |
| Contains `:` (any prefix) | Ollama | `qwen3:4b`, `llama3.1:8b`, **`gpt-oss:120b`**, `hf.co/owner/repo:tag` |
| Otherwise | Ollama | `llama3.1`, `mistral` |

The colon rule is what makes `gpt-oss:120b` correctly route to Ollama instead of OpenAI.

Mixed providers across phases are supported: set `model_qa = "qwen3:4b"` (cheap local Q&A) and `model_citations = "gpt-4o-mini"` (better verbatim citation extraction) and the script initializes a separate client per phase.

### Structured-output mechanism per provider

- **OpenAI**: `client.responses.parse(text_format=PydanticClass)` — native typed parsing.
- **Gemini**: `generate_content(generation_config={"response_mime_type": "application/json", "response_schema": {...}})` — JSON-schema enforcement.
- **Ollama**: `client.chat(format=PydanticClass.model_json_schema())` — JSON-schema enforcement.

---

## Output

### Files written in `output_dir`

| File | When | Contents |
|---|---|---|
| `generated-questions_qa_only.json` | Phase 1 periodic + end | Resume artifact. TEMP- IDs. Carries `context_text`, `citation_extracted`, etc. |
| `generated-questions_with_citations.json` | Phase 2 periodic + end | Resume artifact. TEMP- IDs. Same shape as above plus `full_citation`. |
| **`generated-questions.json`** | After both phases | Sorted, renumbered (`{u_ctx_id}-q-{n}`), `citation_extracted` stripped. |
| **`generated-questions_wo_context.json`** | After both phases | Same as above with `context_text` removed (~46 % smaller). |
| **`generated-questions.csv`** | After both phases | Flat row-per-question. The `citation` column is sourced from `full_citation.citation`. |

### Per-question JSON schema

```json
{
  "question_id": "TBF000011_UKN000-ctx-0-q-0",
  "question": "What is whitetopping in pavement engineering?",
  "answer": "Whitetopping is a rehabilitation technique that ...",
  "full_citation": {
    "citation": "Whitetopping proves to be successful as a rehabilation technique ...",
    "first_sentence": "Whitetopping proves to be successful as a rehabilation technique ...",
    "last_sentence": "Whitetopping proves to be successful as a rehabilation technique ..."
  },
  "question_type": "factual",
  "question_difficulty": "basic",
  "question_element_type": "finding",
  "doc_id": "TBF000011_UKN000",
  "u_ctx_id": "TBF000011_UKN000-ctx-0",
  "u_logic_chunk_id": "TBF000011_UKN000-logic-chunk-0",
  "source_u_chunk_ids": ["TBF000011_UKN000-chunk-1", "TBF000011_UKN000-chunk-2"],
  "artifact_id": "TBF000011_UKN000_chunk_0_art_0",
  "u_artifact_id": "TBF000011_UKN000-ctx-0-art-0",
  "artifact": { "text": "...", "description": "...", "significance": null, "attributes": {...} },
  "context_text": "...",
  "model_qa": "gpt-4o-mini",
  "model_citation": "gpt-4o-mini"
}
```

Field notes:
- The verbatim citation lives at `full_citation.citation`. There is **no** top-level `citation` field; consumers should read `q["full_citation"]["citation"]`.
- `question_element_type` mirrors the artifact category (`finding`, `procedure`, …, plus `summary` for the synthetic element).
- `source_u_chunk_ids` are the original chunk IDs that produced the logical chunk (from `*-logic-ctx.json`).
- `artifact` is the raw artifact dict (text, description, attributes) so a consumer can re-derive prompts if needed.

---

## Logging

- **Console**: level controlled by `--log-level` (default `INFO`).
- **File**: `logs/generate-qa.log`, always at `DEBUG`, 5 MB rotation, zip compression.

`--log-level WARNING` is the cleanest for visual monitoring — only errors + the tqdm progress bar are shown on stderr.

```bash
# Quiet bar only
.venv/bin/python generate-qa.py --config generate-qa.toml --log-level WARNING

# Tail the full debug log while a quiet run is in progress
tail -f logs/generate-qa.log
```

---

## Resume

The script is resumable at both phase boundaries:

| Scenario | What happens on re-run |
|---|---|
| Both intermediates present | Both phases skip-write, only the final renumber + save runs. Sub-second. |
| Only `*_qa_only.json` present | Phase 1 loads + skips (matches by `(u_ctx_id, artifact_id)`). Phase 2 re-extracts all citations. |
| Neither present | Cold run. |
| `generate_qa = false` in TOML, `*_qa_only.json` present | Phase 1 entirely skipped, Phase 2 runs against the loaded intermediate. |
| `extract_citations = false` | Phase 2 entirely skipped. Final files written with `full_citation = null`. |

**Resume keys are TEMP-prefixed.** The intermediate files keep `question_id` in the form `TEMP-{u_ctx_id}-{artifact_id}-{q_idx}` so resume joins remain stable across runs. The final files use the natural-sorted `{u_ctx_id}-q-{n}` form.

---

## Failure modes

### Fatal provider errors (fail-fast)

Permanent errors abort the run in seconds and print a banner at the bottom:

```
============================================================
❌ Execution stopped
Reason: Fatal provider error: 400 API key not valid. ...
============================================================
```

Detected fatal classes per provider:

| Provider | Exceptions treated as fatal |
|---|---|
| OpenAI | `AuthenticationError`, `PermissionDeniedError`, `NotFoundError`, `BadRequestError` |
| Gemini | `InvalidArgument` (400), `Unauthenticated`, `PermissionDenied`, `NotFound`, `FailedPrecondition`, `ResourceExhausted` (quota) |
| Ollama | Error message contains `connection refused`, `model not found`, or `no such model` |

Transient errors (rate limits, timeouts, network glitches) still retry up to 5× before marking a task failed and moving on.

### Common error scenarios

| Symptom | Likely cause | Fix |
|---|---|---|
| `400 API key not valid` | Wrong / missing `GOOGLE_API_KEY` | Update `.env` or env var. |
| `401 Incorrect API key provided: openai-key` | Wrong / missing `OPENAI_API_KEY` | Update `.env` or env var. |
| Routes to OpenAI but model is Ollama (e.g. `gpt-oss:120b`) | `:` was missing from the name | Use the Ollama `name:tag` form. |
| `RuntimeError: Gemini support requires google-generativeai` | Legacy SDK not installed | `pip install google-generativeai` (only when using a `gemini-*` model). |
| `model not found` (Ollama) | Model isn't pulled | `ollama pull <name:tag>` first. |
| Connection refused (Ollama) | Daemon not running | `ollama serve &` (or the desktop app). |

---

## Performance & cost notes

- **Concurrency** is bounded by `max_concurrent_qa` / `max_concurrent_citations`. Default 20 is safe for OpenAI tier-2 and Gemini tier-1; raise on higher tiers or for Ollama.
- **No cost telemetry** is built in — provider dashboards (OpenAI usage page, Google Cloud billing) are the source of truth.
- **No prompt-cache optimization** is configured. OpenAI auto-caches stable prefixes ≥1024 tokens; the current prompts are below that threshold so caching usually does not fire. Cost savings would be modest (~$2-3 on a 50 k-question run at gpt-4o-mini prices). Not currently worth restructuring prompts to chase this.

---

## Examples

### Mix providers across phases

```toml
model_qa = "qwen3:4b"            # cheap local Q&A
model_citations = "gpt-4o-mini"  # better verbatim citation quality
```

### Iterate on Phase 1 prompts only

```toml
generate_qa = true
extract_citations = false        # skip Phase 2 entirely
```

Then later, when you're happy with the Phase 1 output, flip `extract_citations = true` and re-run — Phase 1 will resume from `*_qa_only.json` and Phase 2 will start fresh.

### Force regeneration of citations only

```bash
rm data/nemo_briefs_20260429/qa-gen/generated-questions_with_citations.json
.venv/bin/python generate-qa.py --config generate-qa.toml
```

Phase 1 resumes from the existing `*_qa_only.json` (no LLM calls); Phase 2 re-extracts every citation.

### Smoke test with a tight artifact cap

```toml
include_summary_element = true
max_artifacts_per_ctx = 1   # 1 real artifact + 1 synthetic summary per ctx
```

Roughly halves the corpus while still covering every context.

---

## Known limitations

1. **Citation paraphrase under verbatim prompt** — the citation prompt asks for an exact substring of the context, but models occasionally paraphrase (~1 in 5 spot-checked in smoke runs). Downstream consumers should not assume `citation in context_text` is universally true.
2. **Legacy Gemini SDK dependency** — operators selecting a `gemini-*` model must install `google-generativeai` separately; the project's `reqs.txt` ships `google-genai` (a different SDK).
3. **No prompt-cache tuning** — see "Performance & cost notes" above.
4. **No cost telemetry** — see "Performance & cost notes" above.
5. **Multi-chunk per ctx untested** — Step 2 currently emits one chunk per ctx, so the `\n\n`-joining code path in `build_context_text` is implemented but not exercised at scale.

---

## Related

- **Step 1** (`_nemo.py --sdg-logical`) — see `plans/plan-sdg-logical.md`, `docs/qa-generation.md` § `--sdg-logical`.
- **Step 2** (`extract_artifacts.py`) — see `plans/plan-sdg-logical-step2.md`, `docs/qa-generation.md` § `extract_artifacts.py`.
- **Design plan** for this script — `plans/plan-generate-qa.md`.
- **Formal SRS** for this script — `plans/srs-generate-qa.md`.
- **Reference example** (process-citation flow, untouched) — `examples/qa-generation/generate-data-async2.py`.
