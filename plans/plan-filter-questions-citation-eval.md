# Plan: Citation-based question filter (`filter-questions-citation-eval.py`)

## Context

`generate-qa.py` produces a JSON list of QA records. Phase 2 of that pipeline tries to extract a verbatim citation per question into `full_citation = {citation, first_sentence, last_sentence}`. Two failure modes leave a question with no real citation:

1. **Code-side failure** — after 5 failed LLM retries, `generate-qa.py:754-758` writes the sentinel
   `"Max retries reached. No citation available."` into all three sub-fields.
2. **LLM-side failure** — `prompts/nemo_extract-citation.txt:37-44` instructs the model to return
   `"No relevant citation found"` when it cannot locate a citation in the context.

A QA record with either sentinel (or with a missing/empty citation) is low signal for embedding-model fine-tuning. We need a small, standalone post-processing step that drops those records and writes a clean, filtered file ready for downstream use, without touching the existing pipeline.

## Scope

**In scope**

- New script `filter-questions-citation-eval.py` at repo root (sibling of `generate-qa.py`).
- New config `filter-questions-citation-eval.toml` at repo root (sibling of `generate-qa.toml`).
- Three outputs per run, written next to the input file:
  - `<input_stem>-c-eval.json` — kept records.
  - `<input_stem>-c-eval-dropped.json` — dropped records, each augmented with a `_drop_reason` field.
  - Console + log-file summary: total / kept / dropped, with per-reason breakdown.
- Companion SRS `plans/srs-filter-questions-citation-eval.md`.

**Out of scope**

- No changes to `generate-qa.py`, `_nemo.py`, `extract_artifacts.py`, prompts, or any upstream stage.
- No re-extraction or re-generation of citations. The script is a pure filter.
- No CLI flag for the input path. Input is TOML-only (`--config` selects the TOML).
- No defensive substring/case-insensitive matching for the sentinels — strict equality is enough; the user opted out of the defensive variant.
- No CSV mirror of the filtered set. `generate-qa.py` already ships a CSV path; rerun that on the filtered JSON if needed (out of v1).
- No multi-file batch mode. One input file per invocation.

## Concrete changes

### New: `filter-questions-citation-eval.py`

Top-level standalone script following `generate-qa.py`'s conventions (self-contained helpers, `loguru` logging, `argparse`):

- CLI: `--config` (default `./filter-questions-citation-eval.toml`), `--log-level` (`DEBUG|INFO|WARNING|ERROR`, default `INFO`).
- Logging: `logger.remove()` then `logger.add(sys.stderr, level=args.log_level)` plus rotating file sink at `logs/filter-questions-citation-eval.log` (5 MB rotation, DEBUG level). Mirrors `generate-qa.py:870-873`.
- Config helpers: copy `read_configuration`, `load_json`, `save_to_json` verbatim from `generate-qa.py:132-146` (self-contained style; matches the sibling script). No `aisa.utils.files` dependency, so the script can run with the same minimal env as `generate-qa.py`.
- Sentinels: read from TOML as a list of strings. The first sentinel maps to drop reason `code_sentinel`, the second to `prompt_sentinel`. Reason labels are tied to TOML order so they stay meaningful in logs even if the strings change.
- Classifier: a single function `classify(q: dict, sentinels: list[str]) -> Optional[str]` returning `None` for keep or one of `"code_sentinel" | "prompt_sentinel" | "missing_or_empty"` for drop. The decision tree:
  1. `fc = q.get("full_citation")` — if not a `dict` (None, missing, list, string, etc.) → `missing_or_empty`.
  2. `cit = fc.get("citation")` — if not a `str` or `cit.strip() == ""` → `missing_or_empty`.
  3. `cit == sentinels[0]` (strict equality) → `code_sentinel`.
  4. `cit == sentinels[1]` (strict equality) → `prompt_sentinel`.
  5. Otherwise → `None` (keep).
- Main flow (`main()`):
  1. Parse args, install logging.
  2. `cfg_root = read_configuration(args.config); cfg = cfg_root["filter-questions-citation-eval"]`.
  3. `input_file = cfg["input_file"]; sentinels = list(cfg.get("sentinels", DEFAULT_SENTINELS))`. Ensure `len(sentinels) == 2`; otherwise log error and exit non-zero.
  4. `records = load_json(input_file)` — must be a list; otherwise log error and exit non-zero.
  5. Partition: walk `records`, call `classify`, append to `kept` or `dropped` (with `_drop_reason` injected via in-place set on the existing dict — records are not used after partitioning).
  6. Derive output paths from `input_file` stem: `<stem>-c-eval.json` and `<stem>-c-eval-dropped.json`, both in the same directory as `input_file`.
  7. `save_to_json(kept, kept_path); save_to_json(dropped, dropped_path)`.
  8. Log summary table at INFO: `total`, `kept`, `dropped`, plus a count line per reason (`code_sentinel`, `prompt_sentinel`, `missing_or_empty`). Include kept percentage.
- Exit codes: `0` on success (including empty-result runs, which only emit a WARNING). Non-zero on missing/unreadable input, malformed input shape, or malformed config.

The script is synchronous — no `asyncio`, no LLM calls, no concurrency. Pure CPU/IO over the JSON list.

### New: `filter-questions-citation-eval.toml`

```toml
Title = "Filter Questions by Citation Eval"

[filter-questions-citation-eval]
# Path to the QA JSON produced by generate-qa.py.
# Outputs (sibling of input):
#   <input_stem>-c-eval.json          — kept records
#   <input_stem>-c-eval-dropped.json  — dropped records with _drop_reason
input_file = "./data/nemo_briefs_20260429/qa-gen-cluster/generated-questions_wo_context.json"

# Sentinels treated as low-quality citations (strict equality on full_citation.citation).
# Order matters: index 0 -> code_sentinel reason, index 1 -> prompt_sentinel reason.
# Source of truth:
#   index 0: generate-qa.py:754-758 (retry-exhaustion fallback)
#   index 1: prompts/nemo_extract-citation.txt:37-44 (LLM no-citation fallback)
sentinels = [
  "Max retries reached. No citation available.",
  "No relevant citation found",
]
```

### New: `plans/srs-filter-questions-citation-eval.md`

Companion SRS — see `plans/srs-filter-questions-citation-eval.md`.

### Unchanged

- `generate-qa.py`, `generate-qa.toml`, `prompts/nemo_extract-citation.txt` — read-only references for the sentinel strings.
- `_nemo.py`, `extract_artifacts.py`, `aisa/`, `cfg/nemo.toml` — untouched.
- `reqs.txt` — no new dependencies (only `loguru`, `tomllib`, stdlib; `loguru` is already required by `generate-qa.py`).

## Behavior notes

- **Strict equality, no normalization.** `cit == sentinels[i]` — no `.strip()`, no case-folding, no substring match. The sentinels are produced by code (deterministic) or by an LLM following an explicit instruction with the exact string in the prompt; both reproduce the string verbatim in practice. Borderline cases (model-paraphrased sentinels) would be misclassified as kept; this is acceptable v1 behavior — the user opted out of the defensive variant.
- **Reason labeling pinned to TOML index.** First sentinel → `code_sentinel`; second → `prompt_sentinel`. If the operator reorders the TOML list, the *log labels* swap accordingly. This is intentional: the labels are bound to position so they survive rewording of the sentinel strings.
- **`missing_or_empty` covers everything else.** Any of: `full_citation` is `None` / missing / not a dict / empty dict; `citation` key missing; `citation` value is not a string; `citation` is the empty string or whitespace-only after `.strip()`.
- **`first_sentence` / `last_sentence` are not checked.** The user's filter rule references only `full_citation.citation`. Keeping the rule narrow avoids false drops on partial-fill records that have a real `citation` but malformed `first_sentence`/`last_sentence`.
- **In-place augmentation of dropped records.** `_drop_reason` is added to each dropped record; the kept-set records are written unchanged (byte-equivalent fields). The `_drop_reason` key is prefixed with `_` to flag it as a meta field added by this script.
- **Empty input is allowed.** A zero-length input list emits a WARNING and writes two empty `[]` JSON files. Exit code 0.
- **All-dropped is allowed.** Possible if the input is entirely failure cases. Emits a WARNING.
- **Idempotent.** A second run on the same input overwrites both outputs (file-cache idempotency is not needed here — the input is the only state).

## Verification

End-to-end smoke run on real data:

```bash
python filter-questions-citation-eval.py \
    --config filter-questions-citation-eval.toml \
    --log-level INFO
```

Expected:
- Console summary printed: `total=N, kept=K, dropped=D, reasons={code_sentinel: a, prompt_sentinel: b, missing_or_empty: c}` with `a + b + c == D`.
- Two files appear next to the input: `generated-questions_wo_context-c-eval.json` and `generated-questions_wo_context-c-eval-dropped.json`.

Spot-checks:

- `jq 'length' <input>`, `jq 'length' <kept>`, `jq 'length' <dropped>` → first equals sum of last two.
- `jq '.[].full_citation.citation' <kept> | sort -u | grep -E "Max retries reached|No relevant citation found"` returns nothing (no sentinel passed through).
- `jq '.[]._drop_reason' <dropped> | sort -u` returns a subset of `{"code_sentinel", "prompt_sentinel", "missing_or_empty"}`.
- For each dropped reason, sample one record and confirm the `full_citation.citation` value (or absence) actually matches the reason.

Synthetic edge cases (small fixture JSON of 6 records — one per category):

| # | `full_citation` shape                           | `citation` value                                | Expected reason         |
|---|-------------------------------------------------|-------------------------------------------------|-------------------------|
| 1 | `{citation, first_sentence, last_sentence}`     | "Curing of concrete is important for ..."       | kept                    |
| 2 | `{citation, first_sentence, last_sentence}`     | "Max retries reached. No citation available."   | `code_sentinel`         |
| 3 | `{citation, first_sentence, last_sentence}`     | "No relevant citation found"                    | `prompt_sentinel`       |
| 4 | `null`                                          | —                                               | `missing_or_empty`      |
| 5 | key absent                                      | —                                               | `missing_or_empty`      |
| 6 | `{citation: "  ", first_sentence, last_sentence}` | "  " (whitespace)                             | `missing_or_empty`      |

Run the script pointed at the fixture; assert kept set is `{1}` and dropped reasons match the table.

## Critical files referenced

- `generate-qa.py:88-92` — `CitationResponse` schema (3 fields) — informs the structure of `full_citation`.
- `generate-qa.py:132-146` — `read_configuration` / `load_json` / `save_to_json` helpers to copy verbatim.
- `generate-qa.py:754-758` — origin of the code sentinel (`"Max retries reached. No citation available."`).
- `generate-qa.py:794-802` — where `full_citation` is set on each question via `q["full_citation"] = citation.model_dump()`.
- `generate-qa.py:856-873` — `argparse` + `loguru` setup pattern to mirror.
- `prompts/nemo_extract-citation.txt:37-44` — origin of the prompt sentinel (`"No relevant citation found"`).
- `data/nemo_briefs_20260429/qa-gen-cluster/generated-questions_wo_context.json` — test input.
- `plans/plan-chunk-relevance-filter.md`, `plans/srs-chunk-relevance-filter.md` — plan/SRS file conventions to follow.

## Decisions flagged

- **Sentinels live in TOML, not as code constants.** They originate from two different files (`generate-qa.py` and `nemo_extract-citation.txt`); putting them in TOML makes them visible alongside the operator's run-time choices and lets them evolve without code edits. Code constants would force a script edit on every prompt iteration.
- **Reason labels indexed by TOML position, not by string.** A `dict[str, str]` mapping (e.g. `{"Max retries reached.": "code_sentinel"}`) would re-use the sentinel as a label key — fragile if the strings change. Indexing by position keeps labels stable across rewordings.
- **Strict equality (no `.strip()`, no case-fold, no substring).** Both sentinels are produced verbatim — code-side as a literal, LLM-side as an instruction-followed string. Defensive matching would catch model paraphrases but also risks false drops on legitimate citations that happen to contain the sentinel as a quoted phrase. Strict equality is correct *and* simpler.
- **Output is sibling of input, not configurable.** The user picked this; matches the dataset-as-source-of-truth pattern (the filtered file lives next to the original it derives from).
- **Synchronous script, no `asyncio`.** No I/O concurrency to gain — one JSON read, two JSON writes, pure CPU between. Async would add ceremony with no benefit.
- **No CSV output.** `generate-qa.py:823-849` produces a CSV with the same shape; if needed, run that against the filtered JSON. Avoids duplicating the column list across two scripts.
