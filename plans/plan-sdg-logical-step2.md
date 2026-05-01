# Plan: SDG Pipeline on Logical Chunks — Step 2 (Standalone Artifact Extraction via `langextract`)

Step 2 of the mode-3 (`random_logical`) logical-chunk SDG flow. Step 1 (`plans/plan-sdg-logical.md`) ends after writing `-logic-ctx.json` (one bundle per logical chunk). Step 2 introduces a **standalone script** that consumes the logical chunks and produces `-logic-artifacts.json` via Google's [`langextract`](https://github.com/google/langextract) library, calling OpenAI's `gpt-4o-mini`.

## Architectural framing

Modes 1 (`recursive`) and 2 (`logical`) continue to use the existing in-pipeline `--sdg` flow (`run_sgd_pipeline` in `_nemo.py`) for artifact extraction, QA generation, and evaluation. **Mode 3 forks here**: its downstream stages (artifacts → QA → eval) want fundamentally different shapes — per-chunk extraction, single-segment QA without `multi_hop`/`hop_contexts` machinery — so they live as standalone scripts that share only the chunk directory with the bundled pipeline.

This Step 2 is the first of three planned standalone scripts:
- `extract_artifacts.py` (this plan) — Step 2.
- `generate_qa_logical.py` (future) — Step 3, single-chunk QA, new prompt without multi-hop framing.
- `eval_qa_logical.py` (future) — Step 4, likely a thin wrapper over `nemo_eval`.

Step 1 (`run_sgd_logical_pipeline` in `_nemo.py`) **stays in-pipeline** because it is cheap, has no LLM calls, and only walks the chunk cache. The asymmetry (Step 1 in-pipeline, Steps 2–4 standalone) is deliberate.

`--sdg` + `random_logical` continues to work mechanically (logical chunks get re-bundled by `RecursiveChunker`, multi-hop QA runs across the bundles). It is **supported but discouraged** for mode 3 — use the standalone scripts to get the per-chunk framing this plan is designed for.

## Scope of Step 2

- **Mode 3 only.** Two guards: `_nemo.py:439` tightens `--sdg-logical` to mode 3; `extract_artifacts.py` validates the same at startup.
- One `langextract` call per logical chunk (one entry per `-logic-chunks.json` element). No bundling, no overlap trimming.
- Provider: OpenAI, model `gpt-4o-mini`, key from `.env::OPENAI_API_KEY` (already required by `aisa/gen/providers.py`).
- Domain prompt: pavement engineering — replaces the generic 8-bucket `nemo_artifacts` prompt.
- Script terminates after writing `-logic-artifacts.json`. Steps 3–4 are deferred.
- Existing `--sdg` flow for modes 1, 2, 3 is untouched.

## Naming convention

Step 1 reserved `-logic-artifacts.json` (see `plan-sdg-logical.md` table and SRS FR-8.1). This step actualizes that reservation. Output lives next to `-logic-ctx.json` in `{output_dir}/doc-chunks_{size}_{method}/`. Path derivation: replace the `-logic-chunks.json` suffix with `-logic-artifacts.json` per doc.

## Design choices

1. **Standalone script + importable module.** Logic lives in `aisa/parse/extract.py` (importable, testable, future Step 3 can `from aisa.parse.extract import PavementExtractor`). The repo-root `extract_artifacts.py` is a ~25-line wrapper that does CLI/config/file-walking and delegates per-chunk extraction to the module. Keeps the module reusable; keeps the iteration loop fast.

2. **`PavementExtractor.extract(text, doc_id, chunk_id) -> dict`.** Synchronous. Wraps `lx.extract(text_or_documents=text, prompt_description=prompt, examples=examples, model_id=cfg.model, api_key=cfg.api_key, temperature=cfg.temperature, extraction_passes=cfg.extraction_passes)`. Returns a bucketed-by-class dict in the schema below; `artifact_id` minted per extraction (`f"{doc_id}_chunk_{chunk_id}_art_{ext_idx}"` where `ext_idx` is a 0-based counter across classes within the chunk; matches the underscore-separator convention at `_nemo.py:482`).

3. **Mode-3 guard, two places.**
   - `_nemo.py:439`: tighten `allowed = {"logical", "random_logical"}` → `allowed = {"random_logical"}`. Mode-2 users now see a clear error pointing them at `--sdg`. `--sdg` itself does no method validation today and gains none here, so modes 1 and 2 continue to flow through `--sdg` unchanged.
   - `extract_artifacts.py`: validate `[chunking].method == "random_logical"` immediately after config load; exit with a useful error otherwise.

4. **Pavement-engineering extraction classes** (8, deliberately matching `nemo_artifacts`'s cardinality):
   - `material` — binders, mixes, concretes, aggregates, additives (e.g. "PG 64-22", "Type II cement").
   - `distress` — pavement distress types and severity (e.g. "fatigue cracking", "rutting > 0.5 in").
   - `treatment` — rehabilitation / preservation / maintenance (e.g. "ultrathin whitetopping", "mill-and-fill").
   - `specification` — quantitative requirements (thicknesses, gradations, percentages, ranges).
   - `test_method` — laboratory or field methods (e.g. "Falling Weight Deflectometer", "IRI measurement").
   - `metric` — performance / condition indicators (IRI, ESALs, PCI, structural number).
   - `process` — multi-step procedures (LCCA, scoping field view, design analysis).
   - `reference` — external publications, chapters, sections, tables, figures, appendices, standards (consolidates the 7 classes from `examples/langextract/detect-references.py`; only this class's attributes carry `title`/`source`/`context`).

5. **New prompt `prompts/nemo_logic-artifacts.txt`.** Pavement-domain framing (PennDOT / FHWA / state-DOT documents). Instructs verbatim spans, per-extraction `description` attribute, prefer specific over generic, emit only present classes, no overlapping spans. No `{placeholders}` (langextract supplies the text via `text_or_documents`, not template interpolation).

6. **Few-shot examples** in `aisa/parse/extract.py` as a module-level `PAVEMENT_EXAMPLES: list[lx.data.ExampleData]`. 1–2 examples covering 4–6 of the 8 classes each, so the full vocabulary appears at least once across the example set. Source spans must be verbatim from the example text. Hardcoded in Python — matches `langextract`'s idiom (`examples/langextract/detect-references.py:56`).

7. **On-disk schema for `-logic-artifacts.json`.** Top-level wrapper with `doc_id` (matches `-logic-chunks.json` / `-logic-ctx.json`), `artifacts` array, per-extraction `artifact_id`:
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
           "specification": [
             {
               "artifact_id": "TBF000011_UKN000_chunk_0_art_1",
               "text": "...",
               "char_interval": {"start_pos": 200, "end_pos": 215},
               "attributes": {"description": "..."}
             }
           ],
           "reference": [...]
         }
       }
     ]
   }
   ```
   - `chunk_id` mirrors the corresponding `-logic-chunks.json` entry.
   - `extractions` keys are a **subset** of the 8 classes (omitted when empty — keeps files small for sparse chunks).
   - `artifact_id` is `f"{doc_id}_chunk_{chunk_id}_art_{idx}"` where `idx` increments across all classes within the chunk; globally unique given unique `doc_id`s.
   - `char_interval` is preserved verbatim from `langextract` for downstream evidence offsets.
   - **Two deliberate departures from `-artifacts.json`**: (a) wrapper object with `doc_id` (consistent with Step 1's wrapping convention); (b) per-extraction keys are `artifact_id`/`text`/`char_interval`/`attributes` instead of `text`/`description`/`importance` — `attributes.description` carries what `description` did before. Step 3's QA-gen will need a small `get_fact_blocks` adapter; accepted cost in exchange for `langextract`-native fidelity and traceable IDs.

8. **Idempotency.** Same convention as every other stage:
   ```python
   if Path(out_path).exists() and not overwrite:
       continue
   ```
   The script honors an `--overwrite` CLI flag (default `False`).

9. **Failure isolation.** A `langextract` failure for one chunk SHALL log at `"NLP"` level (chunk_id, error) and continue. The chunk's entry in `artifacts` is still emitted with `"extractions": {}` and an `"error": "<message>"` field. A failure for doc A SHALL NOT abort doc B.

10. **Logging.** Per-doc summary at `"CHUNK"` level: `"<doc>: extracted artifacts from N logical chunks -> .../-logic-artifacts.json"`. Per-chunk timing at `"TIME"` level. Cost telemetry deferred to OQ-2.

11. **Async strategy.** Sequential per chunk, sequential per doc. `lx.extract` is sync; the script is sync. No `asyncio` machinery. If runtime becomes painful, OQ-3 covers concurrency knobs.

## Concrete changes

### New file: `aisa/parse/extract.py` (~150 lines)
- `@dataclass class LXConfig`: `model: str = "gpt-4o-mini"`, `api_key: str | None = None` (resolved from `os.getenv("OPENAI_API_KEY")` if unset), `temperature: float = 0.0`, `extraction_passes: int = 3`, `prompt_name: str = "nemo_logic-artifacts"`, `prompt_lib: str = "./prompts"`.
- `EXTRACTION_CLASSES: list[str]` — frozen 8-class vocabulary; used for output-bucket initialization and validation.
- `PAVEMENT_EXAMPLES: list[lx.data.ExampleData]` — 1–2 module-level examples covering all 8 classes.
- `class PavementExtractor`:
  - `__init__(cfg: LXConfig)` — loads prompt from `Path(cfg.prompt_lib) / f"{cfg.prompt_name}.txt"`. Validates `cfg.api_key`; raises `RuntimeError("OPENAI_API_KEY not set")` otherwise.
  - `extract(text: str, doc_id: str, chunk_id: int) -> dict[str, list[dict]]` — calls `lx.extract`, transforms `result.extractions` into the bucketed schema, mints `artifact_id`s, drops any extractions whose `extraction_class` is outside `EXTRACTION_CLASSES` (logged at `"NLP"` level so we notice prompt drift). Preserves `extraction_text` as `text`, `char_interval` as `{start_pos, end_pos}`, and `attributes`.
  - Raises only on auth/config errors; per-chunk content errors propagate to the caller for failure-isolation handling.
- `def main(cfg: dict, overwrite: bool = False) -> None` — walks `{output_dir}/doc-chunks_{size}_random_logical/`, finds `*-logic-chunks.json`, extracts per chunk, writes `*-logic-artifacts.json` next to it.
- `if __name__ == "__main__":` — argparse + config load + `main(cfg, overwrite)` for direct module invocation.

### New file: `extract_artifacts.py` (~25 lines, repo root)
Thin CLI wrapper:
- argparse: `--cfg` (default `./cfg/nemo.toml`), `--input_dir`, `--output_dir`, `--overwrite`.
- Load TOML, apply CLI overrides to `[general]`.
- Validate `[chunking].method == "random_logical"`; `parser.error(...)` otherwise.
- Call `aisa.parse.extract.main(cfg, overwrite=args.overwrite)`.

### New prompt: `prompts/nemo_logic-artifacts.txt`
- Domain framing (PennDOT / FHWA / state-DOT pavement engineering documents).
- 8-class vocabulary with one-line definitions.
- Instructions: verbatim spans, attach `description`, prefer specific over generic, emit only present classes, no overlapping spans.
- No `{placeholders}` (langextract injects text via `text_or_documents`).

### `_nemo.py` change (one line + error message)
- Line 439: `allowed: set[str] = {"logical", "random_logical"}` → `allowed: set[str] = {"random_logical"}`.
- Line 442: error message updated to `f"--sdg-logical requires [chunking].method == 'random_logical'; got {method!r}. For mode 'logical', use --sdg instead."`.

### `cfg/nemo.toml` — new section
```toml
[langextract]
model = "gpt-4o-mini"
temperature = 0.0
extraction_passes = 3            # langextract recall knob
prompt_name = "nemo_logic-artifacts"
# api_key resolved from .env::OPENAI_API_KEY at runtime
```
Section is optional; missing keys fall back to `LXConfig` defaults.

### `reqs.txt`
Add `langextract>=1.0` (pin to a known-working version after a manual `pip install langextract` probe — currently not installed in the venv, confirmed via `import langextract` failing).

### `docs/qa-generation.md`
- Update `--sdg-logical` section to reflect the mode-3-only guard (Step 1 now rejects mode 2; mode-2 users are pointed at `--sdg`).
- Add a "Step 2: Artifact Extraction (standalone)" subsection documenting `extract_artifacts.py`, the 8-class vocabulary, the schema (incl. `doc_id` wrapper and `artifact_id`), and the OPENAI_API_KEY requirement.
- Note that `--sdg` + `random_logical` is supported but discouraged.
- Mention the schema departure from `-artifacts.json`.

## Out of scope (Steps 3–4)

- **Step 3** (`generate_qa_logical.py`) — single-chunk QA via a new `nemo_logic-qa-gen` prompt. Needs a `get_fact_blocks` adapter for the new bucketed-with-attributes schema. To plan separately.
- **Step 4** (`eval_qa_logical.py`) — likely a thin wrapper over `nemo_eval`. Defer.
- **Cost telemetry through `aisa/gen` decorators** — `langextract` doesn't go through `BaseLLM`, so `ChatResponse`'s cost-accumulation skips it. Plan a tiktoken-based post-hoc estimate as a follow-up (OQ-2).
- **Mode-3 `--prep`** — consuming the eventual `full_logic_sdg_output.json` (or per-doc files directly). Defer until Step 4 lands.

## Verification

1. **Cold run**: with `[chunking].method = "random_logical"` and `-logic-chunks.json` present in `{output_dir}/doc-chunks_{size}_random_logical/`, `python extract_artifacts.py --cfg cfg/nemo.toml` writes `{doc_id}-logic-artifacts.json` for each doc. The file's top-level `doc_id` matches the directory's `{doc_id}` stem and the `doc_id` embedded in `-logic-chunks.json`. The `artifacts` array length equals the `texts` count in `-logic-chunks.json`.
2. **Schema spot-check**: at least one entry has non-empty `material`, one has non-empty `reference`, and `char_interval.start_pos`/`end_pos` are valid integer offsets into the original chunk text (verified by indexing back into `-logic-chunks.json::texts[i].text`).
3. **`artifact_id` integrity**: every extraction has `artifact_id` matching `^.+_chunk_\d+_art_\d+$`; the `chunk_id` parsed out matches the surrounding entry's `chunk_id`; all `artifact_id`s are unique within an output dir (verified via a one-liner aggregator).
4. **Class vocabulary**: every key emitted across all docs is in the 8-class set (no hallucinated classes).
5. **Mode guard, script side**: with `[chunking].method = "recursive"` or `"logical"`, `python extract_artifacts.py` exits with a useful error before any langextract call.
6. **Mode guard, `--sdg-logical` side**: with `[chunking].method = "logical"`, `python _nemo.py --sdg-logical` raises `ValueError` (was previously accepted). With `recursive`, same as before.
7. **`--sdg` regression**: `python _nemo.py --sdg` with `recursive`, `logical`, and `random_logical` still produces `-ctx.json` / `-artifacts.json` / `-qa_pairs.json` / `-qa_eval.json` (modulo LLM nondeterminism). The mode-3-only tightening of `--sdg-logical` does not change `--sdg`.
8. **Idempotency**: second `extract_artifacts.py` run with unchanged inputs and `--overwrite` not set writes nothing (verified via `mtime`) and makes zero `langextract` calls.
9. **Failure isolation**: artificially break one chunk (e.g., truncate to empty) and confirm processing continues; the broken entry has `"extractions": {}` and an `"error"` key.
10. **Provider check**: with `OPENAI_API_KEY` unset, the script fails fast at `PavementExtractor.__init__` with a clear error (not a cryptic 401 mid-run).
11. **Bundled-flow non-interference**: an immediately preceding `--sdg` run leaves no fingerprints in `-logic-artifacts.json`, and `extract_artifacts.py` leaves no fingerprints in `-artifacts.json`.

## Open questions

- **OQ-1** — Output shape: bucketed-by-class (chosen) vs. flat list of extractions with a `class` field per entry. The bucketed form costs us a small loop in `PavementExtractor.extract`; the flat form would cost Step 3 a bigger adapter. Re-evaluate when Step 3 is wired.
- **OQ-2** — Cost / token telemetry. `langextract` does not surface OpenAI usage uniformly. Options: (a) tiktoken-estimate input + count output text, multiply by `gpt-4o-mini` price from `aisa/gen/providers.py::CHAT_MODELS`; (b) hook into `lx`'s underlying client. Defer until we have real runs to motivate the tooling.
- **OQ-3** — Concurrency. Sequential is simple but slow (N chunks × ~3-pass extract). If runtime hurts, add a `concurrent.futures.ThreadPoolExecutor` (sync-friendly) inside `main`. Don't optimize prematurely.
- **OQ-4** — `extraction_passes` default. Example uses 3; that's a 3× cost multiplier for marginal recall on short chunks (~256 tokens). Worth A/B'ing 1 vs. 3 once we have a representative corpus.
- **OQ-5** — Static vs. parameterised prompt. Default static; revisit only if per-corpus class swaps become a need.
- **OQ-6** — Asymmetric provenance: the bundled flow's `-artifacts.json` does not get a `doc_id` wrapper or `artifact_id`s. Intentionally out of scope here; back-porting would invalidate existing caches. If Step 3+ wants symmetric provenance for the bundled flow, that's a separate retrofit task.
- **OQ-7** — Aggregation. `--sdg` writes `full_sdg_output.json` (consumed by `--prep`). Mode 3's parallel `full_logic_sdg_output.json` is a Step 4 question — write it in `eval_qa_logical.py`, or have a future `--prep-logical` walk per-doc files directly?
