# Software Requirements Specification: Model Provenance in Route-B JSON Outputs

**Feature:** Record the LLM model name into the JSON output files of the three Route-B stages
that invoke an LLM but currently record nothing about which model produced the file
(`_nemo.py::evaluate_chunks` → `{doc}-relevance.json`; `_nemo.py::path2chunks` logical-grouping
→ `{doc}-logic-chunks.json`; `extract_artifacts.py::main` → `{doc}-logic-artifacts.json`),
bringing them in line with the already-provenanced `generate-qa.py` and
`self-check/self-check-qa.py`.
**Component:** `nvidia-sdq_custom`
**Version:** 0.1
**Status:** Proposed
**Companion plan:** `plans/plan-model-provenance.md`
**Working notes (background, not authoritative):**
`/Users/igor/.claude/plans/plan-to-add-the-goofy-fiddle.md`

---

## 1. Introduction

### 1.1 Purpose
The Route-B (`random_logical`) pipeline records which LLM produced each output **only for the
last two stages**: `generate-qa.py` writes per-record `model_qa`/`model_citation` into
`qa-gen/generated-questions.json` (`generate-qa.py:621-622`), and
`self-check/self-check-qa.py` writes nested per-record `"model"` into
`self-check-qa-results.json` (`self-check-qa.py:334`, `:383`, `:399`). Those two are flat
record-list / resumable shapes, and per-record provenance was needed there because the user
switched models mid-process.

The **three earlier Route-B stages that also invoke an LLM record nothing** about the model
that produced their file, so an output's provenance (which model graded relevance, grouped
chunks, or extracted artifacts) is unrecoverable after a run. This SRS specifies the
requirements to add a single **per-file header** model field to each of those three outputs.
It is a small, purely-additive source change to two files; **no config, prompt, schema-file, or
dependency change.**

### 1.2 Scope

**In scope**
- Add `"model"` to the `{doc}-relevance.json` header dict written by
  `_nemo.py::evaluate_chunks` (single write site, `_nemo.py:458`), value
  `self.relevance_model`.
- Add `"model"` to the `{doc}-logic-chunks.json` header dict written by
  `_nemo.py::path2chunks` (hybrid/`random_logical` write site, `_nemo.py:339-348`), value
  `self.llm.cfg.model`.
- Add `"span_model"` **and** `"chunk_model"` to the `{doc}-logic-artifacts.json` header dict
  written by `extract_artifacts.py::main` (single write site, `extract_artifacts.py:1065`),
  both value `lx_cfg.model` (the two distinct LLM sub-steps share one model today but may
  diverge — §2).
- A 1-doc Route-B smoke (regenerating cached files via overwrite) that asserts presence,
  asserts **absence** on the two mechanical outputs, and asserts the recorded strings equal
  the configured models.

**Out of scope**
- Any config/TOML, prompt, `ChunkSignals`/Pydantic-schema, or dependency change.
- Per-record (vs per-file) provenance for these three stages — explicitly rejected (§2,
  DECISION 1): each file is written in a single pass with a single model and already has a
  file-level header dict; per-record would N-fold-duplicate one identical string.
- Adding a model field to the two **purely mechanical** Route-B outputs that invoke no LLM —
  explicitly **excluded** (FR-3): `{doc}-chunks.json` (recursive `RecursiveCharacterTextSplitter`
  pre-split) and `{doc}-logic-ctx.json` (`_nemo.py::_build_logical_contexts`, token packing
  only).
- Route A (`--sdg`: in-process `QAGenerator.extract_artifacts` / `generate_qa_pairs` /
  `evaluate_qa_pairs` → `-artifacts.json` / `-qa_pairs.json` / `-qa_eval.json` /
  `full_sdg_output.json`). A separate downstream, not exercised on the `ollama-support`
  branch. Not changed here; noted as possible future work (§10).
- Backfilling provenance into already-cached output files (idempotent file cache — §6, R3).
  New/regenerated runs get the field; the backfill story is documented, not automated.
- Any change to `generate-qa.py` / `self-check/self-check-qa.py` (already provenanced).

---

## 2. Background / Current State

Route-B produces, per `.md` doc: recursive pieces (`{doc}-chunks.json`) → LLM-grouped logical
chunks (`{doc}-logic-chunks.json`) → per-piece relevance scores when the filter is on
(`{doc}-relevance.json`) → 1:1 logical contexts (`{doc}-logic-ctx.json`) → span+chunk artifacts
(`{doc}-logic-artifacts.json`) → questions → self-check scores. Stage I/O is file-based and
idempotent (each stage skips when its output exists and `overwrite` is false; the
`random_logical` cache key is `{doc}-logic-chunks.json`, `_nemo.py:248-252`).

Of the seven Route-B JSON artifacts, **three are produced by an LLM call yet record no model**;
**two are mechanical (no LLM)**; **two already record the model per-record**. This feature closes
the gap for the three LLM-producing files using a per-file header (the existing two record
per-record only because they are resumable flat-record-list files where the user switched
models mid-process — a different shape and a different need).

Audit of every cited locus and every consumer (verified against the working tree this session):

| Locus (file:line) | Binding | Disposition |
|---|---|---|
| `_nemo.py:458` | `files.write_json({"doc_id": doc_id, "scores": scores}, base_out)` — the **single** `-relevance.json` write site; no model field | Add `"model"` (FR-1) |
| `_nemo.py:177-179` | `self.relevance_model = chunk_cfg.get("relevance_model") or self.llm.cfg.model` | The value for FR-1 — exact `model=` arg at `_nemo.py:401` |
| `_nemo.py:401` | `model=self.relevance_model` — the exact string passed to the relevance API call | Confirms FR-1 value is the accurate one |
| `_nemo.py:383-389` | `-relevance.json` cache-hit early-return reads only `cached.get("scores", [])` | Additive sibling key is safe (R1) |
| `_nemo.py:339-348` | hybrid (`random_logical`) `-logic-chunks.json` header dict `{doc_id, parsed_file, texts, images, tables}`; no model field | Add `"model"` (FR-2) |
| `_nemo.py:169` / `chunkers.py:271` | `get_chunker(chunk_cfg, llm)` → `HybridLogicalChunker.self.llm = llm`; same object as `QAGenerator.self.llm` | `self.llm.cfg.model` correct for **both** grouping paths (FR-2.2) |
| `_nemo.py:313-319` (relevance-on) | passes `self.llm` explicitly to `group_kept_pieces` | Same model as relevance-off path |
| `_nemo.py:268-271` (relevance-off) | grouping inside `self.chunker.split()`; `HybridLogicalChunker.llm` IS `self.llm` | Same model — FR-2.2 holds regardless of `relevance_filter` |
| `aisa/gen/chat_llm.py:90,92` | `self.cfg = llm_cfg`; `self.info.name = self.cfg.model` | `self.llm.cfg.model` is the resolved model-name string |
| `extract_artifacts.py:1065` | `_write_json({"doc_id": doc_id, "artifacts": artifacts}, out_path)` — the **single** `-logic-artifacts.json` write site; no model field | Add `"span_model"`+`"chunk_model"` (FR-2.3 / DECISION 2) |
| `extract_artifacts.py:981` | `lx_cfg = LXConfig(**cfg.get("artifact_extraction", {}))` — `lx_cfg` in scope at the write site | The value for both new fields |
| `extract_artifacts.py:620-621` | `class LXConfig: model: str = "gpt-4o-mini"` (default) | Provenance source field |
| `extract_artifacts.py:667,706` (chunk) / `:813,833` (span) | both span and chunk sub-steps use `self.cfg.model` = `lx_cfg.model` | Two fields, one value today; may diverge (DECISION 2) |
| `_nemo.py:282-291` | `-chunks.json` recursive-pieces write (relevance-on path) — mechanical, no LLM | **No** model field (FR-3) |
| `_nemo.py:357-364` | `-chunks.json` non-hybrid write — mechanical | Out of route scope (mode 1/2) |
| `_nemo.py:659` | `files.write_json({"doc_id", "contexts"}, out_path)` — `-logic-ctx.json`, `_build_logical_contexts`, token-packing only | **No** model field (FR-3) |
| `extract_artifacts.py:1006` | consumes `-logic-ctx.json` via `ctx_doc.get("contexts", [])` | Selective key access — additive key safe (R1) |
| `generate-qa.py:492` | consumes `-logic-artifacts.json` via `art_data.get("artifacts", []) or []` | Selective key access — additive key safe (R1) |
| `utils/generate-chunking-report.py:131-132` | consumes `-chunks.json`/`-logic-chunks.json` via `rec_data["texts"]`/`log_data["texts"]` | Selective key access — additive key safe (R1) |
| `generate-qa.py:621-622` | existing per-record `"model_qa"`/`"model_citation"` (two-field) | Precedent for DECISION 2 (two fields) |
| `self-check/self-check-qa.py:334,383,399` | existing nested per-record `"model"` | Precedent for "model" key name; per-record because resumable (not applicable here) |

Verified audit facts (spot-checked this session against the working tree):

- **No consumer anywhere does strict key-allowlist or JSON-schema validation on the three
  target files.** Every reader uses selective `.get(key)`/`[key]` access:
  `-relevance.json` is read only by the `_nemo.py:383-389` cache-hit (reads `scores` only);
  `-logic-chunks.json` is read by `_nemo.py:248-252` (existence-gated cache key; on a
  cache hit reads `.get("texts", [])` only) and
  `utils/generate-chunking-report.py:131-132` (reads `["texts"]` only);
  `-logic-artifacts.json` is read by `generate-qa.py:492` (`art_data.get("artifacts", [])`).
  The `model_validate_json` calls in `generate-qa.py` (`:590,598,727,735`) and
  `extract_artifacts.py` (`:723`) validate **LLM responses**, not these on-disk files. ⇒ The
  new top-level sibling key is provably non-breaking (R1).
- `self.relevance_model` (`_nemo.py:177-179`) is the *exact* string passed as `model=` to the
  relevance call (`_nemo.py:401`) — the most accurate provenance value, not a re-derivation.
- `self.llm.cfg.model` is correct for `-logic-chunks.json` under **both** grouping paths:
  relevance-on passes `self.llm` explicitly to `group_kept_pieces` (`_nemo.py:313-319`);
  relevance-off groups inside `self.chunker.split()` where `HybridLogicalChunker.llm` is the
  same `self.llm` object (`chunkers.py:271`, wired via `get_chunker(chunk_cfg, llm)` at
  `_nemo.py:169`). One value, valid both ways.
- Span and chunk sub-steps in `extract_artifacts.py` both currently use `self.cfg.model`
  (= `lx_cfg.model`): chunk-level at `:667`/`:706`, span-level at `:813`/`:833` (the
  `model=self._model` at `:759` is `_OllamaSpanLM.infer` forwarding langextract's base
  attribute, itself populated from the `model_id=self.cfg.model` at `:813` — not a separate
  source). They are two **distinct LLM call paths** that may be configured to diverge later,
  so two fields
  (`span_model`, `chunk_model`) are recorded even though they hold one value today —
  mirroring the existing `model_qa`/`model_citation` two-field precedent.
- The artifacts model resolves **only** from `[artifact_extraction].model` via
  `LXConfig(**cfg.get("artifact_extraction", {}))` (`extract_artifacts.py:981`). The
  CLAUDE.md note that `[langextract].model` is read when `cfg/nemo.toml` is handed in is
  **stale (a v3→v4 carryover)**: `main()` *raises `KeyError`* if `[langextract]` is present
  without `[artifact_extraction]` (`extract_artifacts.py:962-964`). FR-2.3 therefore sources
  both fields from `lx_cfg.model` only; the SRS does not promise a `[langextract]` path.
- `os` is already imported in `_nemo.py:1` and `_is_ollama_model` already exists at module
  scope (`_nemo.py:124`); no new import is needed for FR-1/FR-2.
- DECISIONS are user-confirmed and fixed: **DECISION 1** per-file header (not per-record);
  **DECISION 2** artifacts gets two fields `span_model`+`chunk_model`.

Design intent only (not current behavior): the `ollama-support` feature
(`plans/srs-ollama-random-logical-pipeline.md`) has landed on this branch, which is why the
line numbers shifted relative to the working-notes draft — every locus above is re-verified
against the current tree.

---

## 3. Functional Requirements

### FR-1 — `{doc}-relevance.json` model provenance (`_nemo.py::evaluate_chunks`)
- **FR-1.1** The single `-relevance.json` write site (`_nemo.py:458`) SHALL write a header dict
  `{"doc_id": doc_id, "model": self.relevance_model, "scores": scores}` (the new `"model"`
  key positioned between `"doc_id"` and `"scores"`).
- **FR-1.2** The recorded value SHALL be `self.relevance_model` (the attribute set at
  `_nemo.py:177-179`, i.e. `chunk_cfg.get("relevance_model") or self.llm.cfg.model`), which is
  byte-identical to the `model=` argument actually sent to the relevance API call
  (`_nemo.py:401`).
- **FR-1.3** No other behavior of `evaluate_chunks` SHALL change: the `_eval_one` per-piece
  loop, `_JSON_BLOCK_RE`/`_SCRATCHPAD_BLOCK_RE` parsing, `RelevanceJudgment` validation,
  per-piece concurrency, the `except → score=1.0` keep-all fallback, the `bins` log line, and
  the `return scores` value SHALL be unchanged.
- **FR-1.4** The `-relevance.json` cache-hit early-return (`_nemo.py:383-389`) SHALL be
  unchanged; it already reads only `cached.get("scores", [])`, so a cached file lacking
  `"model"` continues to work (R1, R3).

### FR-2 — `{doc}-logic-chunks.json` and `{doc}-logic-artifacts.json` model provenance
- **FR-2.1** The hybrid (`random_logical`) `-logic-chunks.json` write
  (`_nemo.py:339-348`) SHALL add a `"model"` key to the header dict, yielding
  `{"doc_id", "model": self.llm.cfg.model, "parsed_file", "texts", "images", "tables"}`.
- **FR-2.2** The recorded value `self.llm.cfg.model` SHALL be correct irrespective of
  `[chunking].relevance_filter`, because both grouping paths use the same `self.llm`
  (relevance-on: explicit arg to `group_kept_pieces`, `_nemo.py:313-319`; relevance-off:
  `HybridLogicalChunker.llm` is `self.llm`, `chunkers.py:271`). The non-hybrid write
  (`_nemo.py:357-364`, modes 1/2) SHALL NOT be changed (out of route scope).
- **FR-2.3** The single `-logic-artifacts.json` write site (`extract_artifacts.py:1065`) SHALL
  write a header dict
  `{"doc_id": doc_id, "span_model": lx_cfg.model, "chunk_model": lx_cfg.model,
  "artifacts": artifacts}`. Two fields SHALL be recorded (DECISION 2): span-level (langextract)
  and chunk-level (`ChunkSignals`) are distinct LLM call paths that may be configured to
  diverge; both equal `lx_cfg.model` today. `lx_cfg` is in scope at the write site
  (`extract_artifacts.py:981`).
- **FR-2.4** No other behavior of `path2chunks`/`evaluate_chunks`/`extract_artifacts.py::main`
  SHALL change (logic-chunk construction, span/chunk extraction, the orchestrator
  failure-isolation `errors.{span,chunk}` contract, the artifacts cache-hit at
  `extract_artifacts.py:1002-1004`). The new keys are header-level siblings only.

### FR-3 — Mechanical outputs explicitly excluded (negative requirement)
- **FR-3.1** `{doc}-chunks.json` (recursive `RecursiveCharacterTextSplitter` pre-split;
  `_nemo.py:282-291` relevance-on, `_nemo.py:357-364` non-hybrid) SHALL **not** gain any
  model field — it is produced by no LLM.
- **FR-3.2** `{doc}-logic-ctx.json` (`_nemo.py::_build_logical_contexts`, write at
  `_nemo.py:659`, token-packing only) SHALL **not** gain any model field — it is produced by
  no LLM.
- **FR-3.3** The §9 smoke SHALL assert these absences (`has("model") | not`) so an accidental
  over-broad edit fails acceptance.

### FR-4 — Additivity / non-breaking invariant
- **FR-4.1** Each change SHALL be a **new top-level sibling key** in an existing header dict;
  no existing key SHALL be renamed, removed, reordered semantically, or have its value
  changed. (The `"model"` key is positioned for readability per FR-1.1/FR-2.1; JSON object
  key order is not semantically significant to any consumer — §2.)
- **FR-4.2** No consumer of the three files SHALL require modification: §2 verified that every
  reader uses selective `.get(key)`/`[key]` access and none does key-allowlist or schema
  validation on these files.
- **FR-4.3** No config/TOML key, prompt file, Pydantic/JSON schema, or dependency SHALL be
  added or changed. `git diff --stat` for the feature SHALL list only `_nemo.py`,
  `extract_artifacts.py`, and the two new `plans/*.md`.

---

## 4. External Interfaces

- **`aisa.utils.files.write_json(obj, path)`** (`_nemo.py:339`, `:458`) — the writer for
  `-logic-chunks.json` and `-relevance.json`; receives one extra dict key. No signature
  change.
- **`extract_artifacts.py::_write_json(obj, path)`** (`extract_artifacts.py:1065`) — the
  writer for `-logic-artifacts.json`; receives two extra dict keys. No signature change.
- **File-handoff boundary** — the only interface that "changes" is the on-disk JSON shape of
  the three files (purely additive header keys, §6). No network, CLI, or Python API change.
- **No new CLI flag.** Regeneration of cached files uses the existing mechanisms:
  `QAGenerator.self.overwrite` (set in `__init__`; no `_nemo.py` config flag exists) or
  deleting the cached file for `_nemo.py`; `extract_artifacts.py --overwrite` for stage 2.

---

## 5. Configuration Schema (effective values)

**No configuration change.** This table records, for the §9 equality assertion, *which
config key the recorded value derives from* — none of these keys is added or modified.

| File | Key | Value (recorded into) |
|---|---|---|
| `cfg/nemo*.toml` | `[chunking].relevance_model` (optional) **or** `[llm].model` | `-relevance.json.model` (= `self.relevance_model`) |
| `cfg/nemo*.toml` | `[llm].model` | `-logic-chunks.json.model` (= `self.llm.cfg.model`) |
| `extract_artifacts*.toml` | `[artifact_extraction].model` | `-logic-artifacts.json.span_model` **and** `.chunk_model` (= `lx_cfg.model`) |

Note: the artifacts model is read **only** from `[artifact_extraction].model`
(`extract_artifacts.py:981`); a `cfg/nemo.toml`-shaped config with `[langextract]` but no
`[artifact_extraction]` raises `KeyError` at `extract_artifacts.py:962-964` (pre-existing
trap, not introduced or fixed here — §2 finding).

---

## 6. Data Flow / Artifacts

Per-doc, under `{output_dir}/doc-chunks_{size}_random_logical/` (chunk dir name carries the
method, `_nemo.py:216-218`). Only the **header dict** of three files changes; record lists are
untouched:

| File | Producer | Header after this feature |
|---|---|---|
| `{doc}-chunks.json` | recursive pre-split (mechanical) | unchanged — **no** model field (FR-3.1) |
| `{doc}-logic-chunks.json` | `path2chunks` LLM grouping | `+ "model": <[llm].model>` (FR-2.1) — also the `random_logical` cache key |
| `{doc}-relevance.json` | `evaluate_chunks` (filter on) | `+ "model": <relevance_model>` (FR-1.1) |
| `{doc}-logic-ctx.json` | `_build_logical_contexts` (mechanical) | unchanged — **no** model field (FR-3.2) |
| `{doc}-logic-artifacts.json` | `extract_artifacts.py` span+chunk LLM | `+ "span_model" + "chunk_model"` (FR-2.3) |
| `qa-gen/generated-questions.json` | `generate-qa.py` | unchanged (already per-record `model_qa`/`model_citation`) |
| `self-check-qa-results.json` | `self-check-qa.py` | unchanged (already per-record nested `model`) |

**Backfill story (idempotent file cache).** Pre-existing cached files do **not** retroactively
gain the field; the stage skips when its output exists and `overwrite` is false (`_nemo.py`
has no config flag — set `QAGenerator.self.overwrite = True` in `__init__` or delete the
cached file; `extract_artifacts.py` has `--overwrite`). The `random_logical` cache key for
stage 1 is `{doc}-logic-chunks.json` (`_nemo.py:248-252`); regenerating it also re-writes
`{doc}-relevance.json` (its `-relevance.json` cache-hit is independent at
`_nemo.py:383-389`, so a stale `-relevance.json` must be deleted/overwritten to gain the
field). Forward runs include the fields; backfill is a documented manual action, not
automated by this feature.

---

## 7. Prerequisites / Assumptions

1. Interpreter: `.venv/bin/python`.
2. The cited loci are at the working-tree line numbers in §2 (re-verified this session, after
   the `ollama-support` feature landed). If the tree drifts, re-locate by symbol
   (`evaluate_chunks` `-relevance.json` write; `path2chunks` hybrid `-logic-chunks.json`
   write; `extract_artifacts.py::main` `_write_json` site), not by line number.
3. Importing `_nemo.py` constructs an `Embedder` default and calls
   `ollama_api.list_models()`; **Ollama must be reachable** at the resolved host for any
   `_nemo.py` run, even `--sdg-logical` which never embeds. `aisa/gen/providers.py` builds
   OpenAI/Google registries at import → **non-empty** `OPENAI_API_KEY`/`GOOGLE_API_KEY` in
   env/`.env` are required just to *import* `_nemo.py`. The §9 smoke runs under bogus-but-
   non-empty keys (zero egress) exactly as `srs-ollama-random-logical-pipeline.md §9`.
4. The §9 smoke uses `[chunking].method = "random_logical"` with `relevance_filter = true`
   (e.g. `cfg/nemo_specs.toml`) so all three target files are produced in one stage-1+stage-2
   pass; the smoke forces regeneration (clean scratch `--output_dir` and/or
   `--overwrite`/`self.overwrite`) so cached files pick up the new fields.
5. A 1-doc subset is sufficient (idempotent file-handoff stages): isolate one `.md` into a
   throwaway dir and override input via `_nemo.py --input_dir` (no `--limit` exists for
   stages 1–2).

---

## 8. Risks & Mitigations

| # | Risk | Mitigation |
|---|---|---|
| R1 | A consumer rejects the new sibling key (strict allowlist / schema validation) ⇒ a downstream stage breaks | §2 verified **every** reader of the three files uses selective `.get`/`[key]` access and none does allowlist/schema validation on these files (the `model_validate_json` calls validate LLM responses, not these files); the new key is provably inert. §9 runs the full Route-B chain through stage 4 to confirm end-to-end |
| R2 | Wrong value recorded (e.g. a re-derived model string drifting from the model actually used) | Each value is the *exact same expression* the stage already passes to its LLM call: `self.relevance_model` is the `model=` arg at `_nemo.py:401`; `self.llm.cfg.model` is the resolved name (`chat_llm.py:90,92`); `lx_cfg.model` is the langextract/chunk model. §9 asserts equality vs the configured keys |
| R3 | Pre-existing cached files never gain the field ⇒ a mixed corpus where old docs lack provenance, silently | Documented backfill story (§6): forward runs include it; to backfill, overwrite/delete. §9 *forces* regeneration (scratch dir + overwrite) so the smoke is not a false pass on a cached file; the `-relevance.json` cache-hit is independent of the `-logic-chunks.json` cache key (delete it too) |
| R4 | Over-broad edit also stamps a model into the mechanical `-chunks.json`/`-logic-ctx.json` (false provenance) | FR-3 makes their *absence* a hard requirement; §9 negative assertions (`has("model") \| not`) fail acceptance if violated |
| R5 | `span_model`/`chunk_model` look redundant (one value today) and a future maintainer collapses them to one field | DECISION 2 is fixed and rationale-recorded (two distinct LLM call paths, chunk-level `extract_artifacts.py:667/706` vs span-level `:813/833`; mirrors the existing `model_qa`/`model_citation` two-field precedent); stated in §1.2 out-of-scope and FR-2.3 |
| R6 | `cfg/nemo_specs.toml` / scratch outputs may be git-untracked (appear under `??`) ⇒ `git diff --stat` (FR-4.3 gate) may not list the smoke's scratch artifacts | FR-4.3 gate is over **source** files only (`_nemo.py`, `extract_artifacts.py`); scratch outputs go to a `/tmp` `--output_dir`, never a tracked path; verify with `git status` |

---

## 9. Acceptance Criteria / Test Plan

**Pre-flight (must hold before running):**
- `git diff --stat` shows only `_nemo.py`, `extract_artifacts.py`, and the two new
  `plans/*.md` modified — **no** config/prompt/schema/dep file (FR-4.3).
- Ollama reachable; non-empty `OPENAI_API_KEY`/`GOOGLE_API_KEY` in env/`.env` for the import
  gate (§7.3). The smoke runs under **bogus-but-non-empty** keys (zero OpenAI/Gemini egress),
  per `srs-ollama-random-logical-pipeline.md §9`.
- A `random_logical` + `relevance_filter = true` config is used (e.g. `cfg/nemo_specs.toml`)
  so all three target files are produced.

**1-doc Route-B smoke (idempotent stages → one doc sufficient; force regeneration so cached
files gain the fields).** Isolate one `.md` into a throwaway dir; use a scratch
`--output_dir` so nothing cached pre-exists:

```
mkdir -p /tmp/model-prov-smoke-in && cp <one>.md /tmp/model-prov-smoke-in/
env OPENAI_API_KEY=sk-bogus GOOGLE_API_KEY=bogus .venv/bin/python _nemo.py \
    --sdg-logical --cfg cfg/nemo_specs.toml \
    --input_dir /tmp/model-prov-smoke-in --output_dir /tmp/model-prov-smoke
env OPENAI_API_KEY=sk-bogus GOOGLE_API_KEY=bogus .venv/bin/python extract_artifacts.py \
    --cfg extract_artifacts_specs.toml \
    --input_dir /tmp/model-prov-smoke/doc-chunks_*_random_logical --overwrite
```

(If a config caches into a real path rather than the scratch dir, also delete the per-doc
`*-relevance.json`/`*-logic-chunks.json`/`*-logic-artifacts.json` or set
`QAGenerator.self.overwrite` — R3.)

Then assert with `jq` over the produced `doc-chunks_*_random_logical/`:

```
D=/tmp/model-prov-smoke/doc-chunks_*_random_logical
jq -e '.model'                       $D/*-relevance.json       # FR-1
jq -e '.model'                       $D/*-logic-chunks.json    # FR-2.1
jq -e '.span_model and .chunk_model' $D/*-logic-artifacts.json # FR-2.3
# negative — mechanical stages MUST NOT gain a model field (FR-3):
jq -e 'has("model") | not'           $D/*-chunks.json
jq -e 'has("model") | not'           $D/*-logic-ctx.json
```

Equality assertions (FR-1.2 / FR-2.2 / FR-2.3, R2): the recorded strings MUST equal the
configured models —
`-relevance.json .model` == `[chunking].relevance_model` (if set) else `[llm].model`;
`-logic-chunks.json .model` == `[llm].model`;
`-logic-artifacts.json .span_model` == `.chunk_model` == `[artifact_extraction].model`.

**Acceptance statement:** the feature is accepted when (a) the three positive `jq -e`
assertions pass on freshly regenerated files, (b) both negative assertions pass, (c) the
equality checks hold against the configured keys, and (d) `git diff --stat` confirms only the
two `.py` files (+ the two `plans/*.md`) changed (FR-4.3).

**Full-run note (optional, recommended).** Idempotent file-handoff stages make the 1-doc
subset sufficient for acceptance. For end-to-end confidence that the additive keys break no
downstream consumer (R1), drive the full Route-B chain
(`_nemo.py --sdg-logical → extract_artifacts.py → generate-qa.py →
self-check/self-check-qa.py`) on the 1-doc subset via the **`pipeline-smoke-runner`** agent
(per `srs-ollama-random-logical-pipeline.md §9`); expected: stages 3–4 complete unchanged
(they read `-logic-artifacts.json` via `art_data.get("artifacts", [])`, unaffected by the
new header keys). This is a confidence check, not a gate.

---

## 10. Future Work (out of scope)

- Per-record provenance for `generate-qa.py`/`self-check-qa.py` is already present; no work.
- Add the analogous header field to **Route A** outputs (`-artifacts.json`,
  `-qa_pairs.json`, `-qa_eval.json`, `full_sdg_output.json` from in-process
  `QAGenerator.extract_artifacts`/`generate_qa_pairs`/`evaluate_qa_pairs`) if Route A is
  revived on this branch.
- Backfill tooling that stamps the configured model into already-cached files without a full
  regeneration (today the only path is overwrite/delete — §6, R3).
- A schema/version field alongside `model` (e.g. prompt-name, temperature) for richer
  provenance, if reproducibility auditing is later required.
- A CI assertion that any LLM-producing Route-B output carries a model field, so a future
  stage cannot regress the provenance contract.
