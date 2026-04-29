# Software Requirements Specification: Question Generation on Logical-Chunk Artifacts (Step 3)

**Feature:** Standalone script `generate-qa.py` + config `generate-qa.toml` + two prompt files. Two-phase async pipeline that consumes mode-3 logical-chunk artifacts and writes per-question records (with verbatim citations) to JSON + CSV. Step 3 of the standalone mode-3 family (Steps 2-4) launched by `extract_artifacts.py` (Step 2).
**Component:** `nvidia-sdq_custom`
**Version:** 0.1 (draft)
**Status:** Implemented
**Companion plan:** `plans/plan-generate-qa.md`

---

## 1. Introduction

### 1.1 Purpose
This SRS defines requirements for a standalone Python script that turns the per-chunk artifact extractions produced by Step 2 (`extract_artifacts.py`) into a corpus of `(question, answer, citation)` triples suitable for downstream embedding-model fine-tuning. The script is adapted from `examples/qa-generation/generate-data-async2.py`; its inputs are remapped from "process citations" to "logical-chunk contexts + langextract artifacts", and its provider routing is reduced to OpenAI / Gemini / Ollama.

### 1.2 Scope

In scope:
- New script `generate-qa.py` at the repo root.
- New config `generate-qa.toml` alongside the script (NOT under `cfg/`).
- Two new prompts: `prompts/nemo_qa-gen-artifact.txt`, `prompts/nemo_extract-citation.txt`.
- Two-phase async pipeline (Phase 1 QA generation, Phase 2 citation extraction) with independent concurrency.
- Per-`(ctx, artifact)` iteration; one LLM call per task in Phase 1, one LLM call per question in Phase 2.
- Provider routing: `gpt-*` → OpenAI Structured Outputs; `gemini-*` → Google Gemini `response_schema`; everything else → Ollama with `format=schema`.
- Resume from prior intermediate files in both phases.
- Periodic save during each phase; final save once.
- Final output: `generated-questions.json` (full per-question records) + `generated-questions.csv` (flattened columns).
- Natural-sort ordering of the final output by `(doc_id, u_ctx_id, artifact_id)` before per-ctx renumbering.
- Symbol replacement (optional, applied to context text in both phases).

Out of scope:
- Mistral / Claude / DeepSeek client paths (dropped from the example).
- Chapter filtering or `source_metadata` `ast.literal_eval` parsing.
- Pushover or any other notification.
- LLM-as-judge evaluation of the generated QA pairs (deferred to Step 4: `eval_qa_logical.py`).
- Integration into `_nemo.py`'s `tasks` dict — Step 3 stays standalone, matching Step 2.
- Cost telemetry through `aisa/gen` decorators; per-call retries delegate to provider SDK defaults plus the script's own 5-try loop.
- Multi-chunk-per-context testing (the code joins `chunks[].text` so it works structurally, but Step 2 currently emits one chunk per ctx).
- Cross-doc parallelism beyond what the per-task semaphore already provides; no doc-level outer pool.

### 1.3 Definitions
- **Context** — one entry of `*-logic-ctx.json::contexts[]`. Identified by `u_ctx_id`. Contains one or more chunks; their `text` fields are concatenated to form the LLM-visible context.
- **Artifact** — one entry of `*-logic-artifacts.json::artifacts[i].extractions[<category>][]`, plus the optional synthetic "summary" artifact constructed from `chunk_signals.summary`.
- **(ctx, artifact) task** — one unit of work in Phase 1: a context paired with an artifact; one LLM call yields 3-5 questions for it.
- **Phase 1** — QA generation. Iterates tasks; appends questions to `all_questions`.
- **Phase 2** — Citation extraction. Iterates questions; updates `citation` and `full_citation` in place.
- **TEMP-ID** — `f"TEMP-{u_ctx_id}-{artifact_id}-{q_idx}"`. Used in intermediate files as the resume key.
- **Final ID** — `f"{u_ctx_id}-q-{n}"` with a per-ctx counter. Assigned by the final renumbering pass; written only to `generated-questions.json` and `generated-questions.csv`.
- **Mode 3** — `[chunking].method = "random_logical"` in `cfg/nemo.toml`. The only mode this script supports as input (it does not validate the mode; it relies on the input files existing in the chunk directory).

### 1.4 References
- `plans/plan-generate-qa.md` — companion design plan.
- `plans/plan-sdg-logical.md`, `plans/srs-sdg-logical.md` — Step 1 (`--sdg-logical`).
- `plans/plan-sdg-logical-step2.md`, `plans/srs-extract-artifacts-v4-chunk-signals.md` — Step 2 (`extract_artifacts.py`) and its v4 output schema (the input to this script).
- `examples/qa-generation/generate-data-async2.py` — reference implementation (process-citation flow, untouched by this SRS).
- `CLAUDE.md` — project conventions and architecture.

---

## 2. Overall Description

### 2.1 Product Perspective
Step 2 (`extract_artifacts.py`) writes one `*-logic-artifacts.json` per doc, carrying per-chunk `extractions` (21-class span-level taxonomy) and `chunk_signals` (summary, topics, terms). Step 1 (`--sdg-logical`) writes the matching `*-logic-ctx.json` with the chunk text. Step 3 reads both files, iterates over `(context, artifact)` pairs, and turns each into 3-5 grounded questions with a verbatim citation extracted from the same context. The output is a flat per-question corpus suitable for biencoder training (the existing `--prep` stage in `_nemo.py` is the bundled-flow analog; a logical-flow `--prep` step is not in this SRS).

### 2.2 User Classes
- **Pipeline operator** — runs `python generate-qa.py --config generate-qa.toml`. Picks a model per phase, tunes concurrency, switches providers via the model name string.
- **Pipeline developer** — extends or tunes the script. Reads provider-routing rules and resume-key contracts before adding state.
- **Downstream consumer** — reads `generated-questions.json` or `generated-questions.csv`. Relies on the per-question schema in §5.4.

### 2.3 Operating Environment
- Python 3.12 (the repo's `.venv` is 3.12; `tomllib` and `from __future__ import annotations` are used).
- `openai==1.91.0`, `ollama==0.5.1`, `pydantic==2.11.7`, `pandas==2.3.0`, `python-dotenv==1.1.0`, `loguru==0.7.3`, `tqdm` (already in `reqs.txt`).
- `google-generativeai` (the legacy SDK) is **optional**; required only when a `gemini-*` model is selected. The project's `reqs.txt` ships `google-genai` (a different package, not used by this script).
- Environment variables: `OPENAI_API_KEY` (required for `gpt-*`); `GOOGLE_API_KEY` (or `GOOGLE_API_KEY_V13`) (required for `gemini-*`); none for Ollama.
- Ollama models additionally require a running Ollama daemon at `--host:--port` (default `localhost:11434`).

### 2.4 Constraints
- The script SHALL be self-contained: no imports from `examples/`. Helpers and Pydantic models are inlined.
- The script SHALL preserve the example's structured-output contracts per provider: `OpenAI.responses.parse(text_format=...)`, Gemini `response_schema=...`, Ollama `format=...`.
- The script SHALL write all phase outputs to a single `output_dir` with the filenames given in TOML.
- The intermediate file IDs (TEMP-) and final IDs (`{u_ctx_id}-q-{n}`) SHALL be stable: a partial Phase 1 followed by a re-run SHALL resume on the same TEMP- keys, and the final renumbering SHALL be deterministic given the same task set.
- Symbol replacement, when enabled, SHALL be applied only to the context text (not the artifact, not the document info, not the question).
- `max_artifacts_per_ctx = 0` SHALL mean "no cap"; values > 0 are treated as a hard cap **before** the synthetic summary element is appended.
- The synthetic summary element SHALL appear at the end of each ctx's task list when `include_summary_element = true` and `chunk_signals.summary.summary` is non-empty.

### 2.5 Assumptions
- Each `*-logic-ctx.json` has a 1:1 sibling `*-logic-artifacts.json` with the same `<doc_id>` prefix. Pairs without a sibling are skipped with a warning.
- Each `u_ctx_id` in `*-logic-ctx.json` matches exactly one `u_ctx_id` in `*-logic-artifacts.json::artifacts[]`. Mismatches are tolerated (the script falls back to `{}` for the artifact entry, yielding an empty artifact list and no synthetic summary unless explicitly avoided).
- `artifact_id` is unique within a `u_ctx_id` — used as part of the Phase 1 resume key.
- The legacy `google.generativeai` SDK is installed by the operator separately when Gemini is needed.

---

## 3. Functional Requirements

### FR-1 Inputs
**FR-1.1** The script SHALL discover doc pairs by globbing `<chunk_dir>/*-logic-ctx.json` and pairing each match by stem with `<doc_id>-logic-artifacts.json` in the same directory.
**FR-1.2** Pairs whose artifacts file is missing SHALL be skipped with a `WARNING` log and SHALL NOT abort the run.
**FR-1.3** Each pair's JSON SHALL be loaded with `load_json` and parsed into Python dicts/lists; no schema validation beyond key presence.

### FR-2 Task construction
**FR-2.1** For each `(doc_id, ctx_path, art_path)`, the script SHALL iterate `ctx_data["contexts"]` in array order. Empty contexts (no chunk text) SHALL be skipped with a `WARNING` log.
**FR-2.2** The matching artifact entry SHALL be looked up by `u_ctx_id`. If absent, an empty dict SHALL be substituted; the synthetic summary element SHALL NOT be emitted in that case.
**FR-2.3** For each context, `extractions[category]` SHALL be flattened across the `artifact_categories` list (in TOML order). Each `(category, artifact)` pair becomes one task.
**FR-2.4** When `max_artifacts_per_ctx > 0`, the flattened list SHALL be truncated to that length **before** the synthetic summary is appended.
**FR-2.5** When `include_summary_element = true` and the artifact entry's `chunk_signals.summary.summary` is non-empty, one synthetic task with `category = "summary"` SHALL be appended after the truncated list.
**FR-2.6** Each task SHALL carry: `doc_id`, `u_ctx_id`, `u_logic_chunk_id` (from `chunks[0]`), `source_u_chunk_ids` (concatenated from all chunks), `context_text`, `doc_info`, `artifact_category`, `artifact_id`, `u_artifact_id`, `artifact` (the raw artifact dict).

### FR-3 Phase 1 — QA generation
**FR-3.1** Phase 1 SHALL bound concurrency at `max_concurrent_qa` via `asyncio.Semaphore`.
**FR-3.2** Each task SHALL substitute its values into `{CONTEXT}`, `{DOCUMENT_INFO}`, `{ARTIFACT_CATEGORY}`, `{ARTIFACT}` in the QA-gen template via `str.replace`.
**FR-3.3** When `replace_symbols = true`, the context text SHALL be passed through `replace_symbols` before substitution.
**FR-3.4** The substituted prompt SHALL be sent to the provider per FR-9 with `temperature = 0.0`.
**FR-3.5** Each task SHALL retry up to 5 times on generic exceptions; `ResourceExhausted` (Gemini quota) SHALL raise `SystemExit` immediately.
**FR-3.6** Each generated question SHALL be packaged into a dict with: TEMP-`question_id`, `question`, `answer`, `citation = None`, `full_citation = None`, `question_type`, `question_difficulty`, `question_element_type = task.artifact_category`, `doc_id`, `u_ctx_id`, `u_logic_chunk_id`, `source_u_chunk_ids`, `artifact_id`, `u_artifact_id`, `artifact`, `context_text`, `model_qa = model`, `model_citation = None`, `citation_extracted = False`.
**FR-3.7** Phase 1 SHALL save the running `all_questions` list to `<output_dir>/<output_qa_file>` every `periodic_save_interval_qa` completed tasks and once at the end of the phase.

### FR-4 Phase 2 — Citation extraction
**FR-4.1** Phase 2 SHALL bound concurrency at `max_concurrent_citations` via `asyncio.Semaphore`.
**FR-4.2** Each pending question (where `citation_extracted == False`) SHALL substitute `{CONTEXT}` and `{QUESTION}` into the citation template.
**FR-4.3** When `replace_symbols = true`, the context text SHALL be passed through `replace_symbols` before substitution.
**FR-4.4** On success, the question dict SHALL be updated in place: `citation = response.citation`, `full_citation = response.model_dump()`, `citation_extracted = True`, `model_citation = model`.
**FR-4.5** On 5 consecutive failures, the dict SHALL be updated with the sentinel triple `{citation: "Max retries reached. No citation available.", first_sentence: ..., last_sentence: ...}` and `citation_extracted = True`.
**FR-4.6** `ResourceExhausted` SHALL raise `SystemExit` immediately.
**FR-4.7** Phase 2 SHALL save the full `all_questions` list to `<output_dir>/<base>_with_citations.<ext>` every `periodic_save_interval_citations` completed questions and once at the end of the phase.

### FR-5 Resume
**FR-5.1** On Phase 1 startup, if `<output_dir>/<output_qa_file>` exists, the script SHALL load it as `existing_questions`.
**FR-5.2** Phase 1 SHALL skip any task whose `(u_ctx_id, artifact_id)` key is already represented in `existing_questions`. New questions SHALL be appended; pre-existing ones SHALL pass through unmodified.
**FR-5.3** On Phase 2 startup, if `<output_dir>/<base>_with_citations.<ext>` exists, citations from prior runs SHALL be merged onto the in-memory questions by `question_id` lookup. The `citation_extracted` flag SHALL be set on merged questions.
**FR-5.4** Phase 2 SHALL skip any question with `citation_extracted == True` after the merge.

### FR-6 Output ordering and final save
**FR-6.1** After Phase 2, `all_questions` SHALL be sorted by `(natural_key(doc_id), natural_key(u_ctx_id), natural_key(artifact_id))`. Natural key is defined as `tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", s or ""))`.
**FR-6.2** A per-`u_ctx_id` counter SHALL assign final `question_id`s of the form `f"{u_ctx_id}-q-{n}"`, with `n` resetting to 0 at each new ctx.
**FR-6.3** The internal `citation_extracted` flag SHALL be removed from each dict before the final save.
**FR-6.4** The final list SHALL be saved to `<output_dir>/<output_file>` and `<output_dir>/<output_csv_file>`. Intermediate files SHALL NOT be overwritten by the final save (they retain TEMP- IDs).

### FR-7 Output schema
**FR-7.1** Per-question JSON record SHALL have the keys defined in §5.4.
**FR-7.2** CSV columns SHALL be (in order): `pandas_index, question_number, question_id, question, answer, citation, question_type, question_element_type, question_difficulty, doc_id, u_ctx_id, u_logic_chunk_id, source_u_chunk_ids, artifact_id, u_artifact_id, artifact, full_citation, model_qa, model_citation`. List/dict cells SHALL be stringified via `str(...)`.

### FR-8 CLI and configuration
**FR-8.1** The script SHALL accept `--config <path>` (default `generate-qa.toml`), `--host <str>` (default `localhost`), `--port <int>` (default `11434`).
**FR-8.2** The TOML SHALL have a `[generate-qa]` section with the keys: `generate_qa, extract_citations, chunk_dir, output_dir, output_file, output_qa_file, output_csv_file, question_generate_prompt, extract_citation_prompt, model_qa, model_citations, max_concurrent_qa, max_concurrent_citations, periodic_save_interval_qa, periodic_save_interval_citations, artifact_categories, include_summary_element, max_artifacts_per_ctx, replace_symbols`.
**FR-8.3** When `replace_symbols = true`, the TOML SHALL also contain one or more `[[generate-qa.symbols]]` array-of-tables with `values: list[str]` and `replace_with: list[str]` (same-index pairs).
**FR-8.4** When `generate_qa = false`, Phase 1 SHALL be skipped; the script SHALL load the existing `<output_qa_file>` and proceed to Phase 2 (or error if the file is missing).
**FR-8.5** When `extract_citations = false`, Phase 2 SHALL be skipped; the final renumber + save SHALL operate on Phase 1's output as-is, and questions retain `citation = null`.
**FR-8.6** Missing required keys SHALL raise `KeyError` at startup before any LLM call.

### FR-9 Provider routing
**FR-9.1** Discriminators: `is_gpt(model)` ⇔ `model.startswith("gpt")`; `is_gemini(model)` ⇔ `model.startswith("gemini")`; `is_ollama_model(model)` ⇔ neither of the above.
**FR-9.2** OpenAI clients SHALL use `OpenAI(api_key=..., timeout=300)` and `client.responses.parse(input=[...], text_format=PydanticClass, temperature=0.0)`.
**FR-9.3** Gemini clients SHALL use `genai.GenerativeModel(model)` after `genai.configure(api_key=...)`. Calls SHALL pass `generation_config={"response_mime_type": "application/json", "response_schema": <JSON Schema dict>, "temperature": 0.0}` and `request_options={"timeout": 300}`.
**FR-9.4** Ollama clients SHALL use `ollama.Client(host=f"http://{host}:{port}")` and `client.chat(messages=[...], model=..., options={"temperature": 0.0}, format=PydanticClass.model_json_schema(), think=False)`.
**FR-9.5** The legacy `google.generativeai` import SHALL be lazy. If it is unavailable, `initialize_client` SHALL raise `RuntimeError` only when a `gemini-*` model is selected; OpenAI / Ollama runs SHALL proceed unaffected.
**FR-9.6** When `model_qa == model_citations`, Phase 1 and Phase 2 SHALL share a single client instance. Otherwise, two clients SHALL be initialized.

### FR-10 Symbol replacement
**FR-10.1** `replace_symbols(text, symbols)` SHALL apply each `(values[i], replace_with[i])` substitution as a verbatim `str.replace`. Empty `values[i]` SHALL be no-ops.
**FR-10.2** Symbol replacement SHALL be applied to the context text only — both at QA generation and at citation extraction. The artifact text, document info, and question text SHALL NOT be transformed.

---

## 4. Non-Functional Requirements

### NFR-1 Backward compatibility
The two prompt files are new; no existing prompt is modified. The new TOML lives alongside the script and does not interact with `cfg/nemo.toml`. `_nemo.py` and `extract_artifacts.py` are untouched. `examples/qa-generation/generate-data-async2.py` is untouched and continues to work for the process-citation flow.

### NFR-2 Determinism
With `temperature = 0.0` and the same provider/model, the LLM-side output is approximately deterministic (subject to provider nondeterminism). The script's ordering is fully deterministic: the natural-sort key (FR-6.1) makes the final JSON / CSV byte-stable across runs given the same task set. Intermediate files use TEMP- IDs and reflect Phase 1 finish order — this is intentional (resume-state, not a user artifact).

### NFR-3 Observability
- `loguru` is configured with a rotating file handler at `logs/generate-qa.log` (5 MB rotation, zip compression, `DEBUG` level).
- Phase boundaries log a banner block; per-phase wall time logs at `INFO`.
- Each periodic save logs at `INFO` with completed/total counts.
- Per-task / per-question success and failure log at `DEBUG` and `ERROR` respectively.
- `tqdm` progress bars are rendered to stderr for both phases.

### NFR-4 Performance envelope
- Per-phase concurrency is operator-tunable via `max_concurrent_qa` and `max_concurrent_citations`.
- Smoke run on the 2-doc fixture (`TBF000011_UKN000` 3 ctx, `TBF000131_UKN000` 30 ctx, `gpt-4o-mini`, `max_concurrent = 20`): Phase 1 produced 1313 questions; Phase 2 extracted 1313 citations in ~75 s.
- No cost telemetry. Per-task tokens are not aggregated (the example's raw-client routing bypasses `aisa/gen` decorators).

### NFR-5 Async safety
- All shared mutable state (`all_questions`) is appended to from the same event loop after `asyncio.as_completed` yields each future. No multi-threading.
- `asyncio.run_in_executor(None, ...)` is used to wrap synchronous SDK calls onto the default thread pool; concurrency is bounded by the per-phase semaphore.
- The OpenAI / Gemini / Ollama clients are documented thread-safe; the script does not mutate shared client state.

### NFR-6 Dependency isolation
- `google-generativeai` SHALL NOT be a hard dependency. The script imports it lazily and gates the import behind `model.startswith("gemini")`.
- All other imports are present in the project's existing `reqs.txt`.

### NFR-7 Input validation surface
- The script does not validate `[chunking].method` because it does not read `cfg/nemo.toml`. The operator is expected to point `chunk_dir` at a directory produced by mode-3 chunking + `extract_artifacts.py`.
- Missing input files at the directory glob produce a `WARNING` (FR-1.2); the run continues with whatever pairs were found.

---

## 5. Interfaces

### 5.1 CLI interface
```text
python generate-qa.py [--config PATH] [--host STR] [--port INT]
```
Defaults: `--config generate-qa.toml`, `--host localhost`, `--port 11434`. No `--overwrite` flag (force-regenerate by deleting the intermediate files).

### 5.2 Python interface (key public surface)
```python
@dataclass
class QATask:
    doc_id: str
    u_ctx_id: str
    u_logic_chunk_id: str
    source_u_chunk_ids: list[str]
    context_text: str
    doc_info: str
    artifact_category: str
    artifact_id: str
    u_artifact_id: str
    artifact: dict
    @property
    def task_key(self) -> str: ...

class GeneratedQuestion(BaseModel):
    difficulty: QuestionDifficulty   # "basic" | "intermediate"
    question: str
    question_type: QuestionType      # "factual" | "conceptual" | "application" | "analysis"
    answer: str

class GeneratedQuestionsResponse(BaseModel):
    questions: list[GeneratedQuestion]

class CitationResponse(BaseModel):
    citation: str
    first_sentence: str
    last_sentence: str

def iter_doc_pairs(chunk_dir: str) -> list[tuple[str, str, str]]: ...
def build_context_text(ctx_entry: dict) -> str: ...
def build_doc_info(doc_id: str, artifact_entry: dict) -> str: ...
def format_artifact(category: str, artifact: dict) -> str: ...
def build_summary_artifact(artifact_entry: dict) -> dict | None: ...
def build_tasks(chunk_dir: str, artifact_categories: list[str],
                include_summary: bool, max_artifacts_per_ctx: int) -> list[QATask]: ...

async def generate_qa_for_task_async(task, template, client, model, symbol_cfg) -> tuple[QATask, list[dict], bool, str | None]: ...
async def run_phase1(tasks, template, client, model, max_concurrent, symbol_cfg,
                     qa_intermediate_path, save_every, existing_questions) -> list[dict]: ...
async def extract_citation_for_question_async(question_text, context_text, template, client, model, symbol_cfg) -> CitationResponse: ...
async def run_phase2(questions, template, client, model, max_concurrent, symbol_cfg,
                     citations_intermediate_path, save_every) -> list[dict]: ...

def save_questions_to_csv(questions: list[dict], path: str) -> None: ...
def main() -> None: ...
```

### 5.3 Configuration interface (TOML)
```toml
Title = "Generate QA"

[generate-qa]
generate_qa = true
extract_citations = true

chunk_dir = "./data/_test/chunk_test-random-logic2/doc-chunks_256_random_logical"

output_dir = "./data/_test/qa-gen"
output_file = "generated-questions.json"
output_qa_file = "generated-questions_qa_only.json"
output_csv_file = "generated-questions.csv"

question_generate_prompt = "./prompts/nemo_qa-gen-artifact.txt"
extract_citation_prompt  = "./prompts/nemo_extract-citation.txt"

model_qa = "gpt-4o-mini"
model_citations = "gpt-4o-mini"

max_concurrent_qa = 20
max_concurrent_citations = 20
periodic_save_interval_qa = 10
periodic_save_interval_citations = 50

artifact_categories = [
  "finding", "issue", "method", "procedure",
  "recommendation", "best_practice", "rationale",
]
include_summary_element = true
max_artifacts_per_ctx = 0

replace_symbols = false
[[generate-qa.symbols]]
values       = ["±", "\\pm", "≥", "≤", "µm"]
replace_with = ["+/-", "+/-", ">=", "<=", "um"]
```

### 5.4 File interface (per-question record)

Per-question JSON shape (final file):
```json
{
  "question_id": "TBF000011_UKN000-ctx-0-q-0",
  "question": "What is whitetopping in pavement engineering?",
  "answer": "...",
  "citation": "...",
  "full_citation": {"citation": "...", "first_sentence": "...", "last_sentence": "..."},
  "question_type": "factual",
  "question_difficulty": "basic",
  "question_element_type": "finding",
  "doc_id": "TBF000011_UKN000",
  "u_ctx_id": "TBF000011_UKN000-ctx-0",
  "u_logic_chunk_id": "TBF000011_UKN000-logic-chunk-0",
  "source_u_chunk_ids": ["TBF000011_UKN000-chunk-1", "TBF000011_UKN000-chunk-2"],
  "artifact_id": "TBF000011_UKN000_chunk_0_art_0",
  "u_artifact_id": "TBF000011_UKN000-ctx-0-art-0",
  "artifact": {"text": "...", "description": "...", "significance": null, "char_interval": {...}, "attributes": {...}},
  "context_text": "...",
  "model_qa": "gpt-4o-mini",
  "model_citation": "gpt-4o-mini"
}
```

Intermediate-file shape: identical except `question_id` is TEMP-prefixed and the `citation_extracted: bool` flag is present.

CSV file shape: `pandas_index, question_number, question_id, question, answer, citation, question_type, question_element_type, question_difficulty, doc_id, u_ctx_id, u_logic_chunk_id, source_u_chunk_ids, artifact_id, u_artifact_id, artifact, full_citation, model_qa, model_citation`.

### 5.5 Environment interface
- `OPENAI_API_KEY` — required when `model_qa` or `model_citations` starts with `gpt`.
- `GOOGLE_API_KEY` (or `GOOGLE_API_KEY_V13`) — required when `model_qa` or `model_citations` starts with `gemini`. The legacy `google-generativeai` package must also be installed.
- No env var required for Ollama; the daemon must be reachable at `http://{host}:{port}`.

---

## 6. Acceptance Criteria

- **AC-1** Running `generate-qa.py` against a chunk directory containing one `*-logic-ctx.json` and one `*-logic-artifacts.json` per doc produces `generated-questions.json`, `generated-questions.csv`, `generated-questions_qa_only.json`, `generated-questions_with_citations.json` under `output_dir`.
- **AC-2** A pair missing the artifacts file produces a `WARNING` log and is skipped; the run continues for the remaining pairs.
- **AC-3** Task count math: for one ctx with N artifacts in the configured categories and `include_summary_element = true`, `build_tasks` yields exactly `min(N, max_artifacts_per_ctx) + 1` tasks (the `+1` is omitted if `chunk_signals.summary.summary` is empty).
- **AC-4** Each per-question record in `generated-questions.json` has the 19 keys listed in §5.4 and no `citation_extracted` key.
- **AC-5** Within `generated-questions.json`, the question-list order satisfies `keys == sorted(keys)` where `keys = [(natural_key(doc_id), natural_key(u_ctx_id), natural_key(artifact_id))]`. (Verifiable by the assertion in `plans/plan-generate-qa.md` §Verification.)
- **AC-6** Within each `u_ctx_id`, the suffix integer of `question_id` (after the last `-q-`) increments from 0 with no gaps.
- **AC-7** Re-running with both intermediate files present skips both phases (zero LLM calls; sub-second wall time).
- **AC-8** Re-running after deleting `generated-questions_with_citations.json` skips Phase 1 (qa-only intermediate present) and re-runs Phase 2 only.
- **AC-9** Switching `model_qa` to `"gemini-2.5-flash"` runs end-to-end provided `google-generativeai` is installed and `GOOGLE_API_KEY` is set; otherwise `initialize_client` raises a clear `RuntimeError`.
- **AC-10** Switching `model_qa` to a local Ollama model (e.g. `"qwen3:4b"`) runs end-to-end provided the daemon is reachable at `--host:--port`.
- **AC-11** Setting `replace_symbols = true` with at least one symbol entry produces context text in the prompt with the substitutions applied (verifiable by inspecting `logs/generate-qa.log` at `DEBUG` level).
- **AC-12** Setting `extract_citations = false` skips Phase 2; the final JSON / CSV have `citation = null` and `full_citation = null` on every record.
- **AC-13** Setting `generate_qa = false` with a stale `generated-questions_qa_only.json` proceeds to Phase 2 only. With no qa-only file present, the script logs a `WARNING` and exits.
- **AC-14** Static smoke check: `python -c "import importlib.util as u; spec = u.spec_from_file_location('g','generate-qa.py'); m = u.module_from_spec(spec); spec.loader.exec_module(m)"` succeeds on a machine without `google-generativeai` installed (validates the lazy import).

---

## 7. Risks and Open Questions

### 7.1 Risks

- **R-1 Citation paraphrase under verbatim prompt.** The Phase 2 prompt asks for an exact substring, but `gpt-4o-mini` paraphrases in ~1 of 5 spot-checked questions on the smoke fixture. Downstream consumers SHALL NOT assume `citation in context_text` is universally true. Mitigation: a post-hoc fuzzy-match validator could be added; deferred.
- **R-2 Legacy Gemini SDK dependency.** Operators selecting a `gemini-*` model must install `google-generativeai` separately. The project's `reqs.txt` ships `google-genai` (a different SDK with a different API surface). If the lazy import fails, the script raises `RuntimeError` only at client init, after task construction has run.
- **R-3 No cost telemetry.** Bypassing `aisa/gen` decorators means token / cost / wall-time per call are not aggregated. Operators must use provider dashboards for cost monitoring. A tiktoken-based post-hoc estimator is a planned follow-up.
- **R-4 Provider-specific failure modes.** OpenAI surfaces 429s through SDK retries; Gemini surfaces `ResourceExhausted` (which the script catches and converts to `SystemExit`); Ollama surfaces connection errors as plain exceptions and falls into the 5-try retry loop. Mixing models across phases (e.g. `model_qa = "gemini-*"`, `model_citations = "gpt-*"`) means the operator must satisfy both providers' env vars and SDKs.
- **R-5 Async-only (no thread pool).** Phase 1 / Phase 2 use a single event loop with `asyncio.run_in_executor` to wrap sync SDK calls. At very high `max_concurrent` values the default thread pool may saturate. Default `20` is safe; values >50 should be tested before production use.
- **R-6 Symbol replacement scope is narrow.** Only the context text is normalized. If a problematic glyph appears in `format_artifact` output (e.g. inside `attributes`), it reaches the model unmodified.

### 7.2 Open Questions (non-blocking)

- **OQ-1** Step 4 (`eval_qa_logical.py`, LLM-as-judge) framing: should it consume `generated-questions.json` directly, or should it expect a re-grouped per-`u_ctx_id` shape? Likely the former, by analogy with `--prep`.
- **OQ-2** Logical-flow `--prep` analog: hard-negative mining + train/test split for the per-question corpus. Unscoped.
- **OQ-3** Multi-chunk-per-context: Step 2 currently emits one chunk per ctx. If that changes, exercise the `\n\n`-join code path with a corpus where it triggers and revisit prompt token budgets.
- **OQ-4** Cost estimator: a tiktoken-based per-call accountant matching the gap noted in `srs-extract-artifacts-v4-chunk-signals.md`.
- **OQ-5** Verbatim-citation enforcement: a Phase 3 validator could re-extract citations whose first run failed `citation in context_text`, with an iteration cap. Defer until paraphrase rate is measured on a larger run.
- **OQ-6** `docs/qa-generation.md` update: append a "Step 3 (`generate-qa.py`)" section paralleling the existing Step 1 / Step 2 sections. Defer until the operator-facing documentation pass is scheduled.
