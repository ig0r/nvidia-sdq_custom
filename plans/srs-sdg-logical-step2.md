# Software Requirements Specification: Standalone Artifact Extraction for Logical Chunks (Mode 3)

**Feature:** Standalone script `extract_artifacts.py` + module `aisa/parse/extract.py` that consumes mode-3 logical chunks and produces `-logic-artifacts.json` via Google's `langextract` library calling OpenAI `gpt-4o-mini`.
**Component:** `nvidia-sdq_custom`
**Version:** 0.1 (draft, Step 2)
**Status:** Proposed

---

## 1. Introduction

### 1.1 Purpose
This SRS defines requirements for the second stage of the mode-3 (`random_logical`) logical-chunk SDG flow: per-chunk artifact extraction implemented as a **standalone script** outside of `_nemo.py`'s pipeline, plus a tightening of Step 1's mode guard so that `--sdg-logical` accepts only mode 3.

### 1.2 Scope

In scope:
- New module `aisa/parse/extract.py` exposing `LXConfig`, `PavementExtractor`, `EXTRACTION_CLASSES`, `PAVEMENT_EXAMPLES`, and a `main(cfg, overwrite)` entry point.
- New thin CLI wrapper `extract_artifacts.py` at the repo root.
- New prompt file `prompts/nemo_logic-artifacts.txt`.
- New optional `[langextract]` section in `cfg/nemo.toml`.
- New dependency `langextract` in `reqs.txt`.
- One-line tightening of `_nemo.py::run_sgd_logical_pipeline`'s `allowed` method set from `{"logical", "random_logical"}` to `{"random_logical"}`, plus an updated error message redirecting mode-2 users to `--sdg`.
- New per-doc output file `{doc_id}-logic-artifacts.json` with the schema defined in §5.3.
- Documentation updates in `docs/qa-generation.md`.

Out of scope:
- Step 3 (`generate_qa_logical.py`) and Step 4 (`eval_qa_logical.py`) — separate SRS revisions.
- Cost telemetry through `aisa/gen` decorators.
- Changes to `--sdg` (`run_sgd_pipeline`), `--prep` (`run_data_prep_pipeline`), `extract_artifacts` (the existing in-pipeline method), `generate_qa_pairs`, `evaluate_qa_pairs`, or any chunker class.
- Changes to existing prompts (`nemo_artifacts`, `nemo_qa-gen`, `nemo_eval`, `nemo_logical-chunk*`).
- Mode-2 (`logical`) extraction. Mode 2 users SHALL run `--sdg` instead.
- Back-porting the `doc_id` wrapper or `artifact_id`s onto the bundled flow's `-artifacts.json`.
- Any aggregate file (`full_logic_sdg_output.json`) — deferred to Step 4.

### 1.3 Definitions
- **Logical chunk** — an output element of `HybridLogicalChunker` (mode `random_logical`), grouped by an LLM into a semantically coherent unit, with `source_chunk_ids` provenance.
- **Mode 3** — `[chunking].method = "random_logical"`. The only mode this feature operates on.
- **Bundled flow** — the existing `--sdg` pipeline (`_nemo.py::run_sgd_pipeline`), unchanged by this feature.
- **Logical flow** — the parallel pipeline composed of `--sdg-logical` (Step 1, in-pipeline) plus the standalone Steps 2–4 scripts.
- **Extraction** — one `lx.data.Extraction`-derived record returned by `langextract`, transformed into this feature's schema.
- **Artifact** — a per-chunk grouping of extractions in this feature's output: `{chunk_id, tokens, extractions, [error]}`.
- **`langextract`** — Google's example-driven span-extraction library (https://github.com/google/langextract).

### 1.4 References
- `plans/plan-sdg-logical-step2.md` — companion implementation plan.
- `plans/plan-sdg-logical.md` and `plans/srs-sdg-logical.md` — Step 1 plan and SRS.
- `examples/langextract/detect-references.py` — reference invocation pattern (Gemini-based; this feature uses OpenAI).
- `CLAUDE.md` — project conventions and architecture.

---

## 2. Overall Description

### 2.1 Product Perspective
The bundled `--sdg` pipeline operates uniformly on chunks regardless of `[chunking].method`. For mode 3, that uniform handling produces multi-segment bundles whose framing diverges from what the user wants: per-logical-chunk single-segment artifacts and QA. This feature establishes the second stage of a parallel, **standalone-script-based** mode-3 pipeline. Step 1 stays in-pipeline (it is cheap and has no LLM calls); Steps 2–4 live as separate scripts so each can be tuned, re-run, and iterated independently.

### 2.2 User Classes
- **Pipeline operator** — runs `python extract_artifacts.py --cfg cfg/nemo.toml` to produce `-logic-artifacts.json` for a corpus already chunked under mode 3.
- **Pipeline developer** — extends the standalone family with subsequent stages (Steps 3–4) and re-uses `PavementExtractor` from `aisa/parse/extract.py`.

### 2.3 Operating Environment
Identical to the bundled flow except for the additional `langextract` dependency. Python 3.x, dependencies in `reqs.txt`, `[chunking].method = "random_logical"` in the resolved config, and `OPENAI_API_KEY` populated in `.env` or the process environment.

### 2.4 Constraints
- The script SHALL operate **only** on mode 3. Modes 1 and 2 SHALL be rejected at startup with a clear error.
- The script SHALL NOT modify, read, or invalidate any artifact written by the bundled flow.
- Output filename derivation SHALL replace the `-logic-chunks.json` suffix with `-logic-artifacts.json`, in the same directory.
- The script SHALL be synchronous. No `asyncio` machinery is permitted; `lx.extract` calls run sequentially per chunk per doc.
- `artifact_id` format SHALL be `f"{doc_id}_chunk_{chunk_id}_art_{idx}"` (underscore-separated, matching `_nemo.py:482`'s `f"{file_name}_chunk_{sid}"` convention) where `idx` is a 0-based counter across all classes for a single chunk.
- Extraction-class vocabulary is fixed at the 8 classes enumerated in §3-FR-5.2; extractions outside this set SHALL be dropped with a logged warning.

### 2.5 Assumptions
- `[chunking].method = "random_logical"` has already been used to produce `{doc_id}-logic-chunks.json` files in `{output_dir}/doc-chunks_{size}_random_logical/`. The script does not invoke chunking itself.
- The user has populated `OPENAI_API_KEY` in `.env` or the environment.
- `langextract` is installed (or can be installed via `pip install -r reqs.txt`).
- The user maintains `cfg/nemo.toml` as the single source of truth for `[general]` and `[chunking]`.

---

## 3. Functional Requirements

### FR-1 CLI surface (`extract_artifacts.py`)
**FR-1.1** The script SHALL accept the following argparse arguments:
- `--cfg` (str, default `./cfg/nemo.toml`).
- `--input_dir` (str, optional; overrides `[general].data_dir`).
- `--output_dir` (str, optional; overrides `[general].output_dir`).
- `--overwrite` (boolean store-true flag, default `False`).
**FR-1.2** With `--help`, the script SHALL print descriptions of all flags and exit 0.
**FR-1.3** With invalid arguments, the script SHALL exit non-zero with a useful error message via `parser.error(...)`.

### FR-2 Configuration loading and validation
**FR-2.1** The script SHALL read the TOML at `--cfg` via `aisa.utils.files.read_toml`.
**FR-2.2** The script SHALL apply `--input_dir` / `--output_dir` overrides to `cfg["general"]["data_dir"]` / `cfg["general"]["output_dir"]` respectively.
**FR-2.3** The script SHALL verify `cfg.get("chunking", {}).get("method") == "random_logical"` after overrides. On violation, it SHALL exit via `parser.error(...)` whose message names the offending value and the required value.
**FR-2.4** The script SHALL verify `OPENAI_API_KEY` is set (either in `cfg.get("langextract", {}).get("api_key")` or `os.environ`). On absence, it SHALL fail at `PavementExtractor.__init__` with `RuntimeError("OPENAI_API_KEY not set")`. The check SHALL fire before any document is processed.
**FR-2.5** The `[langextract]` config section SHALL be optional. Missing keys SHALL fall back to `LXConfig` defaults.

### FR-3 Document discovery
**FR-3.1** The script SHALL compute `chunk_dir = {output_dir}/doc-chunks_{[chunking].chunk_size}_random_logical/` and glob `*-logic-chunks.json` within it.
**FR-3.2** For each match, the script SHALL derive `doc_id` from the filename stem by stripping the `-logic-chunks.json` suffix.
**FR-3.3** If `chunk_dir` does not exist or contains no `*-logic-chunks.json`, the script SHALL log `"NLP"` level and exit 0 (not an error — nothing to do).

### FR-4 Per-chunk extraction
**FR-4.1** For each `-logic-chunks.json`, the script SHALL read the file via `aisa.utils.files.read_json` and extract its `texts` list.
**FR-4.2** For each chunk in `texts`, the script SHALL invoke `PavementExtractor.extract(text=chunk["text"], doc_id=doc_id, chunk_id=chunk["chunk_id"])`.
**FR-4.3** `PavementExtractor.extract` SHALL call `lx.extract(...)` synchronously with `model_id` from `LXConfig`, `api_key` resolved per FR-2.4, `temperature` and `extraction_passes` from `LXConfig`, `prompt_description` loaded from `prompts/{cfg.prompt_name}.txt`, and `examples` set to `PAVEMENT_EXAMPLES`.
**FR-4.4** `PavementExtractor.extract` SHALL transform each returned `lx.data.Extraction` into `{artifact_id, text, char_interval: {start_pos, end_pos}, attributes}`. `artifact_id` SHALL be `f"{doc_id}_chunk_{chunk_id}_art_{idx}"` where `idx` is a 0-based counter across all extractions for that chunk.
**FR-4.5** `PavementExtractor.extract` SHALL group extractions by `extraction_class` into a dict-of-lists keyed by class name. Empty classes SHALL be omitted from the output.
**FR-4.6** `PavementExtractor.extract` SHALL drop any extraction whose `extraction_class` is not in `EXTRACTION_CLASSES` and emit one `"NLP"` log line per drop (chunk_id, offending class).

### FR-5 Extraction-class vocabulary
**FR-5.1** The module SHALL define `EXTRACTION_CLASSES: list[str]` as the immutable list of allowed class names.
**FR-5.2** The vocabulary SHALL be exactly:
`["material", "distress", "treatment", "specification", "test_method", "metric", "process", "reference"]`.
**FR-5.3** `PAVEMENT_EXAMPLES` SHALL contain at least one `lx.data.ExampleData` per class, distributed across 1–2 example documents.

### FR-6 Output writing
**FR-6.1** Output filename SHALL be `{chunk_dir}/{doc_id}-logic-artifacts.json`.
**FR-6.2** The file SHALL contain a JSON object with two top-level keys: `doc_id` (str) and `artifacts` (list).
**FR-6.3** Each element of `artifacts` SHALL conform to:
```
{
  "chunk_id": int,
  "tokens": int,
  "extractions": {<class>: [<extraction>, ...], ...},
  "error"?: str   # present only when extraction failed for this chunk
}
```
**FR-6.4** Each `<extraction>` SHALL conform to:
```
{
  "artifact_id": str,
  "text": str,
  "char_interval": {"start_pos": int, "end_pos": int},
  "attributes": {"description": str, ...class-specific keys}
}
```
**FR-6.5** The file SHALL be serialised via `aisa.utils.files.write_json`.
**FR-6.6** The output SHALL live in the same directory as the corresponding `-logic-chunks.json` (no new directory is created by this feature).

### FR-7 Idempotency
**FR-7.1** Before computing artifacts for a doc, the script SHALL check `Path(out_path).exists() and not overwrite`. On hit, it SHALL skip the doc and emit one `"CHUNK"` log line to that effect.
**FR-7.2** A re-run with no input changes and `--overwrite` not set SHALL incur zero `langextract` calls (verified in AC-8).
**FR-7.3** With `--overwrite`, the script SHALL regenerate every `-logic-artifacts.json` regardless of cache state.

### FR-8 Failure isolation
**FR-8.1** A `langextract` exception during extraction for one chunk SHALL be caught in the script's per-chunk loop. The script SHALL log at `"NLP"` level (doc_id, chunk_id, exception message) and continue to the next chunk.
**FR-8.2** The failed chunk's entry SHALL still be emitted in `artifacts` with `"extractions": {}` and an `"error": str(exc)` field. No `artifact_id`s SHALL be present in such an entry.
**FR-8.3** A failure in one document SHALL NOT abort processing of subsequent documents.
**FR-8.4** Output writes SHALL happen only after all per-chunk extractions for a doc have completed (in-memory list fully built), so a partial doc result is never written to disk.

### FR-9 Logging
**FR-9.1** Per-doc summary at `"CHUNK"` level: `f"{doc_id}: extracted artifacts from N logical chunks -> {out_path}"` after a successful write.
**FR-9.2** Cache hit at `"CHUNK"` level: `f"{doc_id}: cache hit -> {out_path}"`.
**FR-9.3** Per-chunk timing at `"TIME"` level (one line per chunk).
**FR-9.4** Class-vocabulary drift at `"NLP"` level (one line per dropped extraction).
**FR-9.5** Failure isolation at `"NLP"` level (one line per failed chunk).

### FR-10 Mode-3 tightening on `_nemo.py`
**FR-10.1** `_nemo.py:439` SHALL be changed from `allowed: set[str] = {"logical", "random_logical"}` to `allowed: set[str] = {"random_logical"}`.
**FR-10.2** The associated error message at `_nemo.py:441-443` SHALL be updated to: `f"--sdg-logical requires [chunking].method == 'random_logical'; got {method!r}. For mode 'logical', use --sdg instead."`.
**FR-10.3** No other line of `_nemo.py` SHALL be modified by this feature.
**FR-10.4** The existing `--sdg` flag SHALL behave identically before and after this change for all three chunking methods (`recursive`, `logical`, `random_logical`).

### FR-11 Non-interference with bundled flow
**FR-11.1** Running `extract_artifacts.py` SHALL NOT read or write any of: `{doc_id}-ctx.json`, `{doc_id}-artifacts.json`, `{doc_id}-qa_pairs.json`, `{doc_id}-qa_eval.json`, `{doc_id}-sdg.json`, `{root_dir}/full_sdg_output.json`.
**FR-11.2** Running `--sdg` SHALL NOT read or write `{doc_id}-logic-artifacts.json`.

### FR-12 Reserved naming
**FR-12.1** The following filenames remain reserved for future steps (carried over from Step 1's SRS FR-8.1):
- `{doc_id}-logic-qa_pairs.json` (Step 3)
- `{doc_id}-logic-qa_eval.json` (Step 4)
- `{doc_id}-logic-sdg.json` (Step 4)
- `{root_dir}/full_logic_sdg_output.json` (Step 4)

---

## 4. Non-Functional Requirements

### NFR-1 Backward compatibility
Configurations and invocations that do not run `extract_artifacts.py` SHALL behave identically to the pre-change implementation, with one exception: `--sdg-logical` with `[chunking].method = "logical"` now raises `ValueError` (was previously accepted in Step 1). All `--sdg` and `--prep` invocations are unaffected.

### NFR-2 Determinism
With `[langextract].temperature = 0.0`, repeated runs over identical input SHOULD produce identical output, modulo `langextract`'s own retry/sampling behavior across `extraction_passes`. The script's bookkeeping (file walk order, per-doc loop, `artifact_id` indexing) SHALL be deterministic.

### NFR-3 Observability
Every per-doc run SHALL emit at least one `"CHUNK"` log line (success summary or cache hit). Per-chunk timing SHALL be observable at `"TIME"` level. Failures SHALL be observable at `"NLP"` level. Cost telemetry is explicitly out of scope (deferred per OQ-2).

### NFR-4 Performance envelope
The script's runtime is dominated by `lx.extract` calls (one per logical chunk × `extraction_passes`). No specific latency SLA is imposed. Concurrency knobs are deferred (OQ-3).

### NFR-5 Schema parity with `-logic-chunks.json`
Every `chunk_id` in `-logic-artifacts.json::artifacts[*].chunk_id` SHALL appear in the corresponding `-logic-chunks.json::texts[*].chunk_id`, and vice versa. Counts SHALL be byte-equal.

### NFR-6 Dependency isolation
The `langextract` import SHALL live inside `aisa/parse/extract.py`. `_nemo.py`, `aisa/gen/*`, and the rest of `aisa/parse/*` SHALL NOT import `langextract`. Running `--sdg` or `--sdg-logical` SHALL NOT require `langextract` to be installed.

---

## 5. Interfaces

### 5.1 CLI interface
```text
python extract_artifacts.py [--cfg PATH] [--input_dir DIR] [--output_dir DIR] [--overwrite]
```
- Requires `[chunking].method == "random_logical"` in the resolved config.
- Requires `OPENAI_API_KEY` in `.env` or environment.
- No positional arguments.

### 5.2 Python interface
```python
# aisa/parse/extract.py
EXTRACTION_CLASSES: list[str]                       # frozen 8-class vocabulary
PAVEMENT_EXAMPLES: list[lx.data.ExampleData]        # few-shot examples

@dataclass
class LXConfig:
    model: str = "gpt-4o-mini"
    api_key: str | None = None                      # falls back to OPENAI_API_KEY
    temperature: float = 0.0
    extraction_passes: int = 3
    prompt_name: str = "nemo_logic-artifacts"
    prompt_lib: str = "./prompts"

class PavementExtractor:
    def __init__(self, cfg: LXConfig) -> None: ...
    def extract(self, text: str, doc_id: str, chunk_id: int) -> dict[str, list[dict]]: ...

def main(cfg: dict, overwrite: bool = False) -> None: ...
```

### 5.3 File interface
- **Input** (per doc): `{output_dir}/doc-chunks_{size}_random_logical/{doc_id}-logic-chunks.json` — written by `path2chunks` in mode 3. The script reads its `texts` list.
- **Output** (per doc): `{output_dir}/doc-chunks_{size}_random_logical/{doc_id}-logic-artifacts.json`.
- **Output schema**:
  ```json
  {
    "doc_id": "TBF000011_UKN000",
    "artifacts": [
      {
        "chunk_id": 0,
        "tokens": 245,
        "extractions": {
          "material": [
            {
              "artifact_id": "TBF000011_UKN000_chunk_0_art_0",
              "text": "PG 64-22",
              "char_interval": {"start_pos": 145, "end_pos": 153},
              "attributes": {"description": "binder grade used in wearing course"}
            }
          ],
          "reference": [...]
        }
      },
      {
        "chunk_id": 1,
        "tokens": 198,
        "extractions": {},
        "error": "OpenAI rate limit exceeded"
      }
    ]
  }
  ```

### 5.4 Configuration interface
New optional section in `cfg/nemo.toml`:
```toml
[langextract]
model = "gpt-4o-mini"
temperature = 0.0
extraction_passes = 3
prompt_name = "nemo_logic-artifacts"
# api_key resolved from .env::OPENAI_API_KEY at runtime
```
The script SHALL also read `[chunking].method`, `[chunking].chunk_size`, `[general].output_dir`, and `[general].data_dir`.

### 5.5 Environment interface
- `OPENAI_API_KEY` — required. Loaded by `python-dotenv` in `aisa/gen/providers.py`'s import chain (already in place).

### 5.6 Prompt interface
File: `prompts/nemo_logic-artifacts.txt`. Plain text, no `{placeholders}` (langextract supplies the input via `text_or_documents`, not template interpolation). Contains: domain framing, 8-class vocabulary with one-line definitions, instructions on verbatim spans, `description` attribute requirement, no overlapping spans.

---

## 6. Acceptance Criteria

- **AC-1** With `[chunking].method = "random_logical"` and `*-logic-chunks.json` present, `python extract_artifacts.py --cfg cfg/nemo.toml` writes `{doc_id}-logic-artifacts.json` for each doc. The file's top-level `doc_id` matches the directory's `{doc_id}` stem and the `doc_id` in the corresponding `-logic-chunks.json`. The `artifacts` array length equals the `texts` count in `-logic-chunks.json`.
- **AC-2** Each `artifacts` entry has a `chunk_id` and `tokens` byte-equal to the corresponding entry in `-logic-chunks.json::texts[i]`.
- **AC-3** At least one entry across the corpus has non-empty `material`, and at least one has non-empty `reference`. Each emitted `char_interval.start_pos`/`end_pos` is a valid integer offset into the original chunk text (verified by indexing back into `-logic-chunks.json::texts[i].text`).
- **AC-4** Every `artifact_id` matches the regex `^.+_chunk_\d+_art_\d+$`. The `chunk_id` parsed from each ID matches the surrounding entry's `chunk_id`. All `artifact_id`s within a single output dir are unique.
- **AC-5** Every key emitted in `extractions` across all docs is a member of `EXTRACTION_CLASSES`. No hallucinated classes appear.
- **AC-6** With `[chunking].method = "recursive"` or `"logical"`, `python extract_artifacts.py` exits non-zero before any `lx.extract` call, with an error message naming the required mode.
- **AC-7** With `[chunking].method = "logical"`, `python _nemo.py --sdg-logical` raises `ValueError`. The error message redirects the user to `--sdg`.
- **AC-8** A second invocation of `extract_artifacts.py` with unchanged inputs and `--overwrite` not set writes nothing (verified via `mtime`) and makes zero `langextract` calls (verified by token bill / log silence).
- **AC-9** Artificially breaking one chunk (e.g., truncating its text to empty) and rerunning with `--overwrite` produces an entry with `"extractions": {}` and an `"error"` key for that chunk; sibling chunks in the same doc are extracted normally.
- **AC-10** With `OPENAI_API_KEY` unset (and no `[langextract].api_key`), the script fails at `PavementExtractor.__init__` with a clear `RuntimeError`, before any document is processed.
- **AC-11** Running `extract_artifacts.py` immediately after a `--sdg` run does not modify `-ctx.json`, `-artifacts.json`, `-qa_pairs.json`, `-qa_eval.json`, `-sdg.json`, or `full_sdg_output.json`. Running `--sdg` after `extract_artifacts.py` does not modify `-logic-artifacts.json`.
- **AC-12** `python _nemo.py --sdg` continues to work for all three modes (`recursive`, `logical`, `random_logical`) producing the same output filenames as before this feature landed (modulo LLM nondeterminism in content).
- **AC-13** Running `--sdg --sdg-logical` in a single invocation with `[chunking].method = "random_logical"` succeeds: `--sdg` writes the bundled-flow files, `--sdg-logical` writes `-logic-ctx.json`. `extract_artifacts.py` can be run after to write `-logic-artifacts.json`. None of the three steps overwrite each other's outputs.

---

## 7. Risks and Open Questions

### 7.1 Risks

- **R-1** `langextract` may produce extraction classes outside the 8-class vocabulary despite prompt constraints. FR-4.6 mitigates by dropping with a logged warning. If drift is significant, prompt tuning (OQ-5) becomes necessary.
- **R-2** `extraction_passes = 3` triples API cost on short (~256-token) chunks for marginal recall gain. OQ-4 covers an A/B test.
- **R-3** The bundled flow's `-artifacts.json` lacks `doc_id`/`artifact_id` provenance, creating asymmetry across the two flows. OQ-6 covers a possible retrofit.
- **R-4** `langextract` is not installed in the venv (`import langextract` currently fails). The first install may surface API-shape changes vs. the example at `examples/langextract/detect-references.py` (which uses Gemini, not OpenAI). Mitigation: pin a known-working version in `reqs.txt` after manual probe.
- **R-5** Cost is uninstrumented (NFR-3 explicitly defers it). A runaway run on a large corpus could surprise the operator. Mitigation: per-doc summary log lines give a rough proxy via document count; OQ-2 covers proper telemetry.

### 7.2 Open questions (non-blocking)

- **OQ-1** Output shape: bucketed-by-class (chosen) vs. flat list with a `class` field per entry. Re-evaluate when Step 3 is wired.
- **OQ-2** Cost / token telemetry. Defer until a few real runs motivate the tooling.
- **OQ-3** Concurrency. Sequential is the chosen baseline; a `concurrent.futures.ThreadPoolExecutor` upgrade is a follow-up if runtime hurts.
- **OQ-4** `extraction_passes` default of 3 vs. 1. A/B once corpus is settled.
- **OQ-5** Static vs. parameterised prompt. Default static; revisit only if per-corpus class swaps become a need.
- **OQ-6** Retrofitting `doc_id` wrapper / `artifact_id`s onto the bundled flow's `-artifacts.json`. Out of scope here; would invalidate existing caches.
- **OQ-7** Aggregation: `full_logic_sdg_output.json` is a Step 4 question — write it in `eval_qa_logical.py`, or have a future `--prep-logical` walk per-doc files directly?
