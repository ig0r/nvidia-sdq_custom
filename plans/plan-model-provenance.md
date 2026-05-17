# Plan: Record LLM model name in Route-B JSON outputs (per-file header)

**Companion SRS:** `plans/srs-model-provenance.md` (authoritative — this plan refers to its
sections, does not duplicate them)
**Status:** Proposed
**Target:** add a per-file `model` header to the three Route-B outputs produced by an LLM but
currently unprovenanced; purely additive; no config/prompt/schema/dep change
**Working notes (background, not authoritative):**
`/Users/igor/.claude/plans/plan-to-add-the-goofy-fiddle.md`

## Why

The three earlier Route-B LLM stages — relevance filter (`{doc}-relevance.json`), logical
grouping (`{doc}-logic-chunks.json`), artifact extraction (`{doc}-logic-artifacts.json`) —
record nothing about the producing model, unlike the already-provenanced last two stages.
This adds one per-file header field to each. Full rationale, the non-breaking audit, and the
mechanical-output exclusion are in SRS §1–§2.

## Pipeline

Unchanged routes; only three header dicts gain a key:

```
1. _nemo.py --sdg-logical --cfg cfg/nemo_specs.toml
     {doc}-chunks.json        (mechanical) — UNCHANGED, no model field (FR-3)
     {doc}-logic-chunks.json  + "model"  ← Change 2  (FR-2.1)
     {doc}-relevance.json     + "model"  ← Change 1  (FR-1)
     {doc}-logic-ctx.json     (mechanical) — UNCHANGED, no model field (FR-3)
2. extract_artifacts.py --cfg extract_artifacts_specs.toml
     {doc}-logic-artifacts.json + "span_model" + "chunk_model"  ← Change 3 (FR-2.3)
3. generate-qa.py            — UNCHANGED (already per-record provenanced)
4. self-check/self-check-qa.py — UNCHANGED (already per-record provenanced)
```

## Decisions (user)

- **DECISION 1 — per-file header, NOT per-record** (fixed). Each of the three stages writes
  its file in a single pass with a single model, and each file already has a file-level
  header dict. Per-record (as qa-gen/self-check do, because they are resumable flat
  record-lists and the user switched models mid-process) would only N-fold-duplicate one
  identical string here.
- **DECISION 2 — artifacts gets two fields** `span_model` + `chunk_model` (fixed). Span-level
  (langextract) and chunk-level (`ChunkSignals`) are two distinct LLM call paths
  (chunk-level `extract_artifacts.py:667/706` vs span-level `:813/833`) that may be
  configured to diverge; both = `lx_cfg.model` today. Mirrors the existing
  `model_qa`/`model_citation` two-field precedent (SRS §2, FR-2.3).
- Scope is **only** the three LLM-producing files; mechanical outputs and Route A are out
  (SRS §1.2, FR-3, §10).
- No config/prompt/schema/dependency change (SRS FR-4.3).

## Change 1 — `_nemo.py::evaluate_chunks`: `-relevance.json` model

Single write site, `_nemo.py:458`:

```python
# before
files.write_json({"doc_id": doc_id, "scores": scores}, base_out)
# after
files.write_json(
    {"doc_id": doc_id, "model": self.relevance_model, "scores": scores},
    base_out,
)
```

`self.relevance_model` is set at `_nemo.py:177-179`
(`chunk_cfg.get("relevance_model") or self.llm.cfg.model`) and is the exact `model=` arg sent
to the relevance call at `_nemo.py:401` — the most accurate value (SRS FR-1.2). Nothing else
in `evaluate_chunks` changes; the cache-hit early-return (`_nemo.py:383-389`) already reads
only `cached.get("scores", [])` and is left as-is (SRS FR-1.3/FR-1.4, R1/R3).

## Change 2 — `_nemo.py::path2chunks`: `-logic-chunks.json` model

Hybrid (`random_logical`) write, `_nemo.py:339-348` — add `"model"` to the header dict:

```python
files.write_json(
    {
        "doc_id": doc_id,
        "model": self.llm.cfg.model,
        "parsed_file": parsed_file,
        "texts": logic_chunks,
        "images": images,
        "tables": tables,
    },
    cache_path,
)
```

`self.llm.cfg.model` is the resolved model-name string (`aisa/gen/chat_llm.py:90,92`),
correct for **both** grouping paths: relevance-on passes `self.llm` explicitly to
`group_kept_pieces` (`_nemo.py:313-319`); relevance-off groups inside `self.chunker.split()`
where `HybridLogicalChunker.llm` is the same `self.llm` (`chunkers.py:271`, wired via
`get_chunker(chunk_cfg, llm)` at `_nemo.py:169`) — one value, valid both ways (SRS FR-2.2).
The non-hybrid write (`_nemo.py:357-364`, modes 1/2) and the recursive `-chunks.json` write
(`_nemo.py:282-291`) are **not** touched (SRS FR-2.2, FR-3.1).

## Change 3 — `extract_artifacts.py::main`: `-logic-artifacts.json` two model fields

Single write site, `extract_artifacts.py:1065` (`lx_cfg` in scope from `:981`):

```python
# before
_write_json({"doc_id": doc_id, "artifacts": artifacts}, out_path)
# after
_write_json(
    {
        "doc_id": doc_id,
        "span_model": lx_cfg.model,
        "chunk_model": lx_cfg.model,
        "artifacts": artifacts,
    },
    out_path,
)
```

Two fields per DECISION 2; both `lx_cfg.model` (`LXConfig.model`,
`extract_artifacts.py:620-621`, sourced from `[artifact_extraction].model` at `:981`). The
orchestrator failure-isolation (`errors.{span,chunk}`) and the artifacts cache-hit
(`extract_artifacts.py:1002-1004`) are unchanged (SRS FR-2.4).

## Critical files

- `/Users/igor/dev/llm/pavement-gpt/nvidia-pipeline/nvidia-sdq_custom/_nemo.py`
  (Change 1 — `:458`; Change 2 — `:339-348`; values: `self.relevance_model` `:177-179`,
  `self.llm.cfg.model` via `chat_llm.py:90,92`)
- `/Users/igor/dev/llm/pavement-gpt/nvidia-pipeline/nvidia-sdq_custom/extract_artifacts.py`
  (Change 3 — `:1065`; `lx_cfg` `:981`, `LXConfig.model` `:620-621`)
- **No config/prompt/schema/dependency file changes** (SRS FR-4.3). `git diff --stat` for the
  feature lists only these two `.py` files plus the two new `plans/*.md`.

## Prerequisites

1. `.venv/bin/python` interpreter.
2. Loci at the §2 working-tree line numbers (re-verified this session, post-`ollama-support`;
   re-locate by symbol if drifted — SRS §7.2).
3. Ollama reachable; non-empty `OPENAI_API_KEY`/`GOOGLE_API_KEY` in env/`.env` for the
   `_nemo.py` import gate (§7.3). Smoke runs under bogus-but-non-empty keys (zero egress).
4. A `random_logical` + `relevance_filter = true` config (e.g. `cfg/nemo_specs.toml`) so all
   three target files are produced; force regeneration (scratch `--output_dir` and/or
   `--overwrite`/`self.overwrite`) so cached files gain the fields (SRS §6, R3).
5. A 1-doc subset (idempotent stages) isolated into a throwaway dir + `_nemo.py --input_dir`.

## Verification

Per SRS §9 (do not duplicate — summary). 1-doc Route-B smoke into a scratch `--output_dir`
with forced regeneration:

1. `_nemo.py --sdg-logical --cfg cfg/nemo_specs.toml --input_dir <1-doc dir>
   --output_dir /tmp/model-prov-smoke` (bogus-but-non-empty keys → zero egress).
2. `extract_artifacts.py --cfg extract_artifacts_specs.toml
   --input_dir /tmp/model-prov-smoke/doc-chunks_*_random_logical --overwrite`.
3. `jq -e` assertions over `doc-chunks_*_random_logical/`: `.model` present in
   `-relevance.json` and `-logic-chunks.json`; `.span_model and .chunk_model` in
   `-logic-artifacts.json`; **negative** `has("model")|not` on `-chunks.json` and
   `-logic-ctx.json` (FR-3); equality of recorded strings vs configured
   `[chunking].relevance_model`/`[llm].model`/`[artifact_extraction].model`.
4. `git diff --stat` shows only the two `.py` files (+ two `plans/*.md`) — no
   config/prompt/schema/dep (SRS FR-4.3).
5. Optional end-to-end confidence: drive the full Route-B chain
   (`--sdg-logical → extract_artifacts → generate-qa → self-check`) on the 1-doc subset via
   the **`pipeline-smoke-runner`** agent — stages 3–4 must complete unchanged (they read
   `-logic-artifacts.json` via `art_data.get("artifacts", [])`, unaffected). Confidence
   check, not a gate (SRS §9 full-run note).

## Risks

Full table in SRS §8. Biggest: **R1** — a consumer rejects the new sibling key; largely
retired by the SRS §2 audit (every reader uses selective `.get`/`[key]`; no
allowlist/schema validation on these files) and the §9 end-to-end run. R2–R6 (wrong value,
stale cache, over-broad edit, `span_model`/`chunk_model` redundancy, untracked scratch) are
mitigated as recorded in SRS §8 — not restated here.
