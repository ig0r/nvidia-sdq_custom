# Software Requirements Specification: Citation-based question filter

**Feature:** Add a standalone post-processing script that filters QA records produced by `generate-qa.py`, dropping records whose `full_citation.citation` field is missing/empty or matches one of two known "no citation" sentinels — one set by code on retry exhaustion, the other instructed by the citation-extraction prompt to the LLM. Kept records are written to `<input_stem>-c-eval.json`; dropped records (with `_drop_reason` annotation) to `<input_stem>-c-eval-dropped.json`; a per-reason summary is logged to console and rotating log file.

**Component:** `nvidia-sdq_custom`
**Version:** 0.1 (draft)
**Status:** Proposed
**Companion plan:** `plans/plan-filter-questions-citation-eval.md`

---

## 1. Introduction

### 1.1 Purpose

This SRS defines requirements for a small, self-contained Python script that filters out low-quality QA records based solely on citation state, after `generate-qa.py` has run. The goal is to provide downstream stages (embedding fine-tuning, evaluation harness) with a JSON file that is guaranteed to contain only records with a real, model-extracted verbatim citation. The script does not call any LLM, does not require model credentials, and does not modify upstream artifacts.

### 1.2 Scope

In scope:
- A new script `filter-questions-citation-eval.py` at the repo root.
- A new TOML config `filter-questions-citation-eval.toml` at the repo root.
- Three outputs per invocation: kept JSON, dropped JSON (with `_drop_reason`), and a logged summary.
- A companion `plans/plan-filter-questions-citation-eval.md` and `plans/srs-filter-questions-citation-eval.md` (this document).

Out of scope:
- Any modification to `generate-qa.py`, `_nemo.py`, `extract_artifacts.py`, prompts, or `aisa/`.
- LLM-based re-extraction of failed citations.
- Citation-quality grading beyond the two sentinels and emptiness (e.g. citation-context similarity scoring).
- Multi-file batch mode (one input JSON per invocation).
- A CLI flag for the input path; input is TOML-only.
- A CSV mirror of the filtered set.
- Defensive substring / case-insensitive sentinel matching.

### 1.3 Definitions

- **QA record** — one element of the JSON list produced by `generate-qa.py`. A dict with at least `question_id`, `question`, `answer`, and (when Phase 2 ran) `full_citation`.
- **`full_citation`** — dict with three string fields `{citation, first_sentence, last_sentence}` populated from `CitationResponse.model_dump()` at `generate-qa.py:799`. May be `None` if Phase 2 didn't run for the record.
- **Code sentinel** — the literal string `"Max retries reached. No citation available."` set into all three `full_citation` sub-fields at `generate-qa.py:754-758` after the LLM citation-extraction call fails 5 retries.
- **Prompt sentinel** — the literal string `"No relevant citation found"` instructed by `prompts/nemo_extract-citation.txt:37-44` as the LLM's no-citation fallback.
- **Drop reason** — one of `code_sentinel`, `prompt_sentinel`, `missing_or_empty`.
- **Strict equality** — `==` comparison, no normalization (no `.strip()`, no case-folding, no substring).
- **Kept record** — a QA record for which `classify(...)` returns `None`. Written to `<input_stem>-c-eval.json` unchanged.
- **Dropped record** — a QA record for which `classify(...)` returns a non-`None` reason. Written to `<input_stem>-c-eval-dropped.json` with `_drop_reason: <reason>` injected in-place.
- **Summary** — INFO-level log line(s) showing total, kept, dropped, and per-reason counts.

### 1.4 References

- `plans/plan-filter-questions-citation-eval.md` — companion implementation plan.
- `generate-qa.py:88-92, 132-146, 754-758, 794-802, 856-873` — source of citation schema, IO helpers, sentinel literal, citation assignment, and CLI/logging conventions to mirror.
- `prompts/nemo_extract-citation.txt:37-44` — source of the prompt sentinel.
- `plans/plan-chunk-relevance-filter.md`, `plans/srs-chunk-relevance-filter.md` — sibling plan/SRS documents whose conventions this SRS follows.
- Test input: `data/nemo_briefs_20260429/qa-gen-cluster/generated-questions_wo_context.json`.

---

## 2. Overall Description

### 2.1 Product Perspective

`generate-qa.py` runs a two-phase pipeline: Phase 1 generates QA pairs from artifacts; Phase 2 extracts a verbatim citation per question. Phase 2 has two failure escapes — a code-side retry-exhaustion sentinel and an LLM-instructed no-citation sentinel — both of which leave the record with a `full_citation` whose `citation` field carries the sentinel string. Records may also have `full_citation = None` (Phase 2 was skipped) or empty/whitespace `citation`. None of these records are useful for embedding fine-tuning, where the citation is supposed to ground the answer in a verbatim text span.

This filter inserts a single post-processing step between `generate-qa.py` and downstream consumers. It reads the input JSON list, partitions records into kept and dropped sets via a deterministic classifier, and writes the two sets to sibling JSON files. The classifier is closed-form (no LLM, no I/O beyond config + input + outputs) and runs in milliseconds per thousand records.

### 2.2 User Classes

- **Pipeline operator** — runs `python filter-questions-citation-eval.py` after `generate-qa.py`. Configures `input_file` and (rarely) the sentinel list in TOML. Reads the summary to gauge data quality.
- **Pipeline developer** — extends the classifier (e.g. tighter rules), tunes drop-reason categories, integrates the script into a Slurm pipeline. Updates the SRS when the rules change.
- **Downstream consumer** — reads `<input_stem>-c-eval.json` for fine-tuning / evaluation. Sees a smaller list with guaranteed citation presence.

### 2.3 Operating Environment

Identical to `generate-qa.py`'s minimum environment: Python 3.11+ (for `tomllib`), `loguru` for logging. No `aisa.*` imports, no model credentials, no network access. The script runs identically on Slurm, on a developer laptop, and inside CI.

### 2.4 Constraints

- The script SHALL only read its input file, the TOML config, and stdin/argv.
- The script SHALL only write its two output JSON files and the rotating log under `logs/`.
- The script SHALL NOT modify the input file.
- The classifier SHALL use strict equality (`==`) for sentinel comparison; no `.strip()`, no case-folding, no substring matching.
- The script SHALL exit non-zero if the input file does not exist, cannot be parsed as JSON, or is not a top-level JSON array. The script SHALL exit non-zero if the TOML config is missing required keys (`input_file`) or if `sentinels` is not a list of exactly two strings.
- The script SHALL exit zero on success, including all-empty and all-dropped runs.
- The script SHALL NOT raise on a record missing optional fields (`question`, `answer`, etc.); only `full_citation`-related shape is inspected by the classifier.
- The script SHALL be synchronous. No `asyncio`, no threads.

### 2.5 Assumptions

- The input file is a JSON list at the top level (not an object with a wrapper key). This matches `save_to_json` output at `generate-qa.py:143-146` for the `all_questions` list.
- Each record is a `dict`. The script does not validate record structure beyond the `full_citation` field (extra/missing other keys are OK).
- The two sentinels are reproduced verbatim in the data when their respective failure paths fire. Empirically true: the code-side sentinel is a literal at `generate-qa.py:754-758`; the LLM-side sentinel is shown verbatim in the prompt's example block and instruction at `prompts/nemo_extract-citation.txt:39-44`.
- The operator runs the script from the repo root so that `./logs/` and the relative `input_file` path in the TOML resolve correctly. Same convention as `generate-qa.py`.

---

## 3. Functional Requirements

### FR-1 — Config loading

- **FR-1.1** The script SHALL accept `--config <path>` (default `./filter-questions-citation-eval.toml`) and `--log-level <DEBUG|INFO|WARNING|ERROR>` (default `INFO`) via `argparse`.
- **FR-1.2** The script SHALL load the TOML config via `tomllib.load`, mirroring `generate-qa.py:132-135`.
- **FR-1.3** The script SHALL read its config from the `[filter-questions-citation-eval]` table.
- **FR-1.4** The script SHALL require `input_file: str` in the config; absence SHALL cause a non-zero exit with an ERROR log line.
- **FR-1.5** The script SHALL read `sentinels: list[str]` from the config; if absent, SHALL fall back to the two literals defined as `DEFAULT_SENTINELS` in code (matching the source-of-truth strings). If present, SHALL validate `len(sentinels) == 2` and each is a non-empty string; otherwise non-zero exit.

### FR-2 — Input loading

- **FR-2.1** The script SHALL open `input_file` and parse it as JSON via `json.load`, mirroring `generate-qa.py:138-140`.
- **FR-2.2** The script SHALL verify the parsed value is a `list`; otherwise log an ERROR and exit non-zero.
- **FR-2.3** The script SHALL log an INFO line `Loaded N records from <input_file>`.
- **FR-2.4** Empty input list SHALL be accepted; the script logs a WARNING and proceeds to write two empty `[]` files.

### FR-3 — Classification

- **FR-3.1** The script SHALL define `classify(q: dict, sentinels: list[str]) -> Optional[str]` returning `None` for keep or one of `"code_sentinel"`, `"prompt_sentinel"`, `"missing_or_empty"` for drop.
- **FR-3.2** `classify` SHALL evaluate the rules in this order, returning on the first match:
  1. If `q.get("full_citation")` is not a `dict` → `"missing_or_empty"`.
  2. If `cit = full_citation.get("citation")` is not a `str` or `cit.strip() == ""` → `"missing_or_empty"`.
  3. If `cit == sentinels[0]` (strict equality) → `"code_sentinel"`.
  4. If `cit == sentinels[1]` (strict equality) → `"prompt_sentinel"`.
  5. Otherwise → `None` (keep).
- **FR-3.3** The classifier SHALL inspect only the `full_citation.citation` field. The fields `first_sentence`, `last_sentence` SHALL NOT influence the decision.

### FR-4 — Partitioning and annotation

- **FR-4.1** The script SHALL iterate over the input list once, calling `classify` per record.
- **FR-4.2** Records for which `classify` returns `None` SHALL be appended to a `kept: list[dict]`, unchanged.
- **FR-4.3** Records for which `classify` returns a non-`None` reason SHALL be appended to `dropped: list[dict]` with `q["_drop_reason"] = reason` set in-place before append.

### FR-5 — Output

- **FR-5.1** Output paths SHALL be derived from the input path: `kept_path = <input_dir>/<input_stem>-c-eval.json`, `dropped_path = <input_dir>/<input_stem>-c-eval-dropped.json`.
- **FR-5.2** The script SHALL write `kept` to `kept_path` via `save_to_json` (UTF-8, indent=2, `ensure_ascii=False`), mirroring `generate-qa.py:143-146`.
- **FR-5.3** The script SHALL write `dropped` to `dropped_path` via the same `save_to_json` helper.
- **FR-5.4** Both writes SHALL create the parent directory if missing (`os.makedirs(... exist_ok=True)`); in normal operation it already exists (sibling of input).
- **FR-5.5** Existing files at the output paths SHALL be overwritten without prompt or backup.

### FR-6 — Summary

- **FR-6.1** The script SHALL log an INFO summary line containing `total=<N>`, `kept=<K>`, `dropped=<D>`, and `kept_pct=<K/N as percent, 1 decimal>`.
- **FR-6.2** The script SHALL log one INFO line per drop reason with the count, even if the count is 0: `code_sentinel=<a>`, `prompt_sentinel=<b>`, `missing_or_empty=<c>`. `a + b + c` SHALL equal `D`.
- **FR-6.3** The script SHALL log INFO lines naming the two output paths actually written.

### FR-7 — Logging

- **FR-7.1** The script SHALL configure `loguru` with `logger.remove()` followed by `logger.add(sys.stderr, level=args.log_level)` and `logger.add("logs/filter-questions-citation-eval.log", rotation="5 MB", compression="zip", level="DEBUG")`. Mirrors `generate-qa.py:870-873`.
- **FR-7.2** The script SHALL `os.makedirs("logs", exist_ok=True)` before adding the file sink.

### FR-8 — Exit codes

- **FR-8.1** Exit `0` on successful run (including empty input, all-kept, all-dropped).
- **FR-8.2** Exit non-zero (`1` or higher) on:
  - Missing/unreadable config file.
  - Missing required config keys.
  - Malformed `sentinels` config (length ≠ 2 or non-string elements).
  - Missing/unreadable input file.
  - Input is not a top-level JSON list.

---

## 4. Non-Functional Requirements

### NFR-1 — Performance

- **NFR-1.1** The script SHALL process at least 100,000 records per second on a single core (filtering is `O(N)` over a small dict; the dominant cost is JSON parse/serialize).
- **NFR-1.2** Memory footprint SHALL be `O(N)` (one full pass; both kept and dropped lists held in memory before write). For the expected input scale (≤100K records), this is well under 1 GB.

### NFR-2 — Reliability

- **NFR-2.1** The classifier SHALL be deterministic. Same input → same output.
- **NFR-2.2** The script SHALL be idempotent. Repeated runs on the same input produce the same output files (modulo `os` modification time).
- **NFR-2.3** The script SHALL NOT swallow exceptions silently. Any unexpected error SHALL be logged at ERROR with traceback and propagate as a non-zero exit.

### NFR-3 — Maintainability

- **NFR-3.1** The script SHALL be self-contained — copies of `read_configuration`, `load_json`, `save_to_json` are inlined from `generate-qa.py:132-146`. No `aisa.*` imports.
- **NFR-3.2** The classifier function SHALL be testable in isolation (pure function over `(dict, list[str]) -> Optional[str]`).
- **NFR-3.3** Drop-reason labels SHALL be string constants defined at module scope so callers can reference them (e.g. for tests).

### NFR-4 — Compatibility

- **NFR-4.1** The script SHALL work on Python 3.11+ (uses `tomllib`).
- **NFR-4.2** The script SHALL NOT introduce new entries to `reqs.txt`.

---

## 5. Acceptance Criteria

- **AC-1** `filter-questions-citation-eval.py` exists at the repo root and is executable via `python filter-questions-citation-eval.py`.
- **AC-2** `filter-questions-citation-eval.toml` exists at the repo root with a `[filter-questions-citation-eval]` table containing `input_file` and `sentinels` (length 2).
- **AC-3** Running the script with `--config filter-questions-citation-eval.toml` against the test input produces `generated-questions_wo_context-c-eval.json` and `generated-questions_wo_context-c-eval-dropped.json` in the same directory as the input.
- **AC-4** `len(kept) + len(dropped) == len(input)` — verified by `jq 'length'` on all three files.
- **AC-5** Every record in `kept` has a `full_citation.citation` that is a non-empty string and not equal to either configured sentinel.
- **AC-6** Every record in `dropped` has a `_drop_reason` field whose value is one of `{"code_sentinel", "prompt_sentinel", "missing_or_empty"}`.
- **AC-7** The summary on the console reports `total`, `kept`, `dropped`, and per-reason counts; the per-reason counts sum to `dropped`.
- **AC-8** The synthetic 6-record fixture (Verification table in the plan) yields the expected partition: kept set is `{record #1}`, dropped reasons are `{2: code_sentinel, 3: prompt_sentinel, 4-6: missing_or_empty}`.
- **AC-9** Running the script twice in a row with no changes produces byte-identical output files (idempotency).
- **AC-10** Running with a missing input file produces a non-zero exit and an ERROR log line naming the missing path. No output files are created.
- **AC-11** Running with a `sentinels` list of length 1 or 3 in the TOML produces a non-zero exit and an ERROR log line naming the malformed key.
- **AC-12** The script does not import from `aisa.*` and does not require `OPENAI_API_KEY` / `GOOGLE_API_KEY`. Verified by `grep -E "from aisa|OPENAI_API_KEY|GOOGLE_API_KEY" filter-questions-citation-eval.py` returning nothing relevant.

---

## 6. Out-of-scope follow-ups

- **OQ-1** Optional CSV mirror of the kept set (for parity with `generate-qa.py`'s CSV path). Skipped in v1.
- **OQ-2** Citation-context grounding check (does `citation` actually appear verbatim in the source `context_text`?). Would require keeping the chunk text alongside the question — currently dropped from `*_wo_context.json`.
- **OQ-3** Length-based heuristics (drop citations under N characters or N sentences). Not requested in v1; the existing prompt already asks for ≥2-3 sentences.
- **OQ-4** Defensive sentinel matching (substring + case-insensitive). Explicitly opted out by the user; revisit only if sentinel-paraphrasing is observed in real data.
- **OQ-5** Multi-file batch mode (glob over `qa-gen*/generated-questions*.json`). Skipped in v1; operators can drive this from a shell loop.

---

## 7. Decisions flagged

- **D-1 Sentinels in TOML, not constants.** Rationale in plan §"Decisions flagged".
- **D-2 Reason labels indexed by TOML position.** Stable across sentinel-string rewordings.
- **D-3 Strict equality only.** Aligns with the deterministic origin of both sentinels; the user explicitly opted out of defensive matching.
- **D-4 In-place `_drop_reason` injection.** Underscore prefix flags it as a meta field; in-place mutation is safe because records are not reused after partitioning.
- **D-5 Synchronous, no async.** No I/O concurrency to gain.
- **D-6 No CSV in v1.** Re-running `generate-qa.py`'s CSV-saver against the kept JSON is a separate, explicit step if needed.
