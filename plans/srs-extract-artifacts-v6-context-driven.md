# Software Requirements Specification: Extract-Artifacts v6 — Context-Driven Input

**Feature:** Switch `extract_artifacts.py`'s primary input from `*-logic-chunks.json` (per-chunk) to `*-logic-ctx.json` (per-context), aligning the script's iteration unit with `generate-qa.py`. The text passed to both span and chunk calls is the joined chunk text per context (formula identical to `generate-qa.py::build_context_text`). The `*-logic-ctx.json` sidecar becomes a hard requirement; the v5 silent-fallback path is removed. The artifacts file's per-record shape reorders to put `u_ctx_id` first; for 1:1 contexts (current `_build_logical_contexts` behavior) the records are byte-equivalent to v5 modulo the `u_logic_chunk_id` source change (now read from sidecar instead of derived).
**Component:** `nvidia-sdq_custom`
**Version:** 0.6 (draft)
**Status:** Proposed
**Companion plan:** `plans/plan-extract-artifacts-v6-context-driven.md`

---

## 1. Introduction

### 1.1 Purpose
Eliminate the per-chunk vs per-context iteration split between `extract_artifacts.py` and `generate-qa.py`. v6 makes `extract_artifacts.py` consume the same primary input file (`*-logic-ctx.json`), iterate per context, and use the same context-text formula. The change is structural, not algorithmic — for the current 1:1 `_build_logical_contexts` output, model inputs are unchanged, so quality and cost are unchanged. The win is consistency, single-source-of-truth ID minting, and forward compatibility with future N:1 bundling.

### 1.2 Scope

In scope:
- `main()`: glob `*-logic-ctx.json` instead of `*-logic-chunks.json`; iterate the contexts list; build text via the `generate-qa.py` formula; submit one extraction call per context to the within-doc thread pool.
- Remove the v5 silent-fallback path that derives `u_ctx_id` when the sidecar is missing. Sidecar is now required per doc.
- `u_logic_chunk_id` now sourced from the sidecar (`chunks[0].u_logic_chunk_id`) rather than derived from doc_id + chunk_id.
- `tokens` now sourced from the context-level aggregate (`ctx.tokens`) rather than the chunk's own count. Same value for 1:1.
- Module docstring + `--input_dir` CLI help updated to reflect the input-file change.
- `docs/qa-generation.md` updated.

Out of scope:
- v5 contract changes that aren't directly affected (per-chunk wrapper field set, `errors` dict shape, span-level entry shape, `chunk_signals` shape, mode-3 guard, idempotency, threading model). v6 keeps all of these from v5.
- Renaming the `chunk_id` field on the wrapper. Kept as `chunks[0].chunk_id` for 1:1 backwards compat; documented as deprecated under future N:1.
- Adding offsets-relative-to-individual-chunks for span `char_interval` under N:1. Today's 1:1 case has joined text == chunk text, so offsets are unchanged.
- Changing the extraction call signature or the v5 thread-pool model.
- Denormalizing `source_u_chunk_ids` onto the artifact record. Provenance to recursive chunks is preserved via join: `u_ctx_id` → `-logic-ctx.json::contexts[*].chunks[*].source_u_chunk_ids`. Adding it on the artifact would duplicate normalized provenance with no concrete consumer benefit.

### 1.3 Definitions
- **v5** — prior state. Reads `*-logic-chunks.json` per chunk; loads `*-logic-ctx.json` only as a sidecar for `u_ctx_id` lookup; falls back to a derived id if the sidecar is missing.
- **v6** — state introduced by this SRS. Reads `*-logic-ctx.json` per context; computes context text via the `generate-qa.py` formula; sidecar required.
- **Context** — one entry of `ctx_doc["contexts"]`. Carries `u_ctx_id`, a `chunks` list (≥1), and aggregate `tokens`. Today's `_build_logical_contexts` produces 1:1 (one chunk per context); N:1 is a future possibility.
- **Per-context record** — one entry of the v6 `-logic-artifacts.json` `artifacts` list. Replaces v5's per-chunk record.

### 1.4 References
- v5 plan + SRS: `plans/plan-extract-artifacts-v5-concurrency.md`, `plans/srs-extract-artifacts-v5-concurrency.md`.
- `generate-qa.py:273-286` (`iter_doc_pairs`), `:289-292` (`build_context_text`), `:380-428` (per-context iteration).
- `_nemo.py::_build_logical_contexts` at `_nemo.py:594-621` (the producer of `*-logic-ctx.json`).

## 2. Overall Description

### 2.1 Product Perspective
Today the extraction pipeline reads chunks; the QA pipeline reads contexts. They happen to agree because contexts are 1:1 with chunks. v6 unifies the iteration unit at the *context* level — same as `generate-qa.py` — so that both stages share a single source of truth and the contract holds whether contexts are 1:1 or N:1. The output schema's per-record shape changes minimally: `u_ctx_id` becomes the primary key (reordered to first), and `u_logic_chunk_id` / `tokens` are now sourced from the sidecar instead of derived. `chunk_id` is preserved for 1:1 ergonomics.

### 2.2 User Classes
- **Pipeline operator** — must run `_nemo.py --sdg-logical` (or chained `--chunk-only --sdg-logical`) before `extract_artifacts.py`. The "v5 silent fallback" workflow no longer applies.
- **Pipeline developer** — extends or tunes either stage. Now reads one input shape across both.
- **Downstream consumer** — `generate-qa.py` continues to join by `u_ctx_id` against the artifacts file. v5 → v6 join behavior is unchanged.

### 2.3 Operating Environment
Identical to v5: Python 3.11+, `langextract==1.2.1`, `openai==1.91.0`, `pydantic==2.11.7`, `loguru`, `python-dotenv`, `OPENAI_API_KEY`. No new dependencies.

### 2.4 Constraints
- The per-doc top-level wrapper (`{doc_id, artifacts: [...]}`) SHALL NOT change.
- Per-record shape SHALL be `{u_ctx_id, u_logic_chunk_id, chunk_id, tokens, extractions, chunk_signals, errors}`. Field semantics in §1.2.
- Per-context records SHALL appear in the same order as `ctx_doc["contexts"]` (preserved by the order-preserving futures iteration from v5 FR-3.3, applied to context units instead of chunk units).
- Span-level entries inside `extractions` SHALL retain the v4 6-key shape (`{artifact_id, u_artifact_id, text, description, significance, char_interval, attributes}`). The `u_artifact_id` is `f"{u_ctx_id}-art-{ext_idx}"` (unchanged from v5 patch).
- `chunk_signals` SHALL retain the v4 Pydantic-validated shape.
- `errors` SHALL retain the v4 `{span: str|null, chunk: str|null}` shape.
- The mode-3 guard, idempotency, threading model, and OpenAI `max_retries=5` from v5 are preserved verbatim.
- The `*-logic-ctx.json` sidecar SHALL be the only primary input. The v5 fallback path SHALL be removed.

### 2.5 Assumptions
- `_nemo.py --sdg-logical` (or equivalent) has been run before `extract_artifacts.py`, producing `*-logic-ctx.json` per doc.
- `_build_logical_contexts` continues to populate each chunk entry with `u_logic_chunk_id`, `chunk_id`, `text`, `tokens`, `source_u_chunk_ids`. (Verified by reading `_nemo.py:594-621`.)
- Each `ctx_entry["chunks"]` has at least one element (empty contexts are filtered with an `"NLP"` log).

## 3. Functional Requirements

### FR-1 Primary input
**FR-1.1** `main()` SHALL glob `chunk_dir.glob("*-logic-ctx.json")` (replacing v5's `*-logic-chunks.json`).
**FR-1.2** If no `*-logic-ctx.json` files are found, the script SHALL log "No `*-logic-ctx.json` under {chunk_dir}; nothing to do" at `"NLP"` level and exit cleanly (mirrors v5's empty-input behavior, just with the new suffix).
**FR-1.3** `doc_id` SHALL be derived as `chunks_path.name.replace("-logic-ctx.json", "")`.
**FR-1.4** The output filename SHALL remain `{doc_id}-logic-artifacts.json`.

### FR-2 Context iteration and text construction
**FR-2.1** For each `ctx_entry` in `ctx_doc.get("contexts", [])`, the script SHALL:
- Skip if `chunks = ctx_entry.get("chunks", [])` is empty (with `"NLP"` log naming the `u_ctx_id`).
- Compute `text` as `"\n\n".join(c.get("text", "") for c in chunks if c.get("text")).strip()`. If the resulting string is empty, skip with `"NLP"` log.
- Submit one extraction task per context to the within-doc thread pool.

**FR-2.2** The text-construction formula SHALL be byte-identical to `generate-qa.py:289-292::build_context_text`.

### FR-3 Per-record shape
**FR-3.1** Each record in the per-doc `artifacts` list SHALL have keys `{u_ctx_id, u_logic_chunk_id, chunk_id, tokens, extractions, chunk_signals, errors}` — exactly these seven, no more, no fewer.
**FR-3.2** `u_ctx_id` SHALL be `ctx_entry["u_ctx_id"]`.
**FR-3.3** `u_logic_chunk_id` SHALL be `chunks[0].get("u_logic_chunk_id", "")` from the sidecar (NOT derived from doc_id + chunk_id).
**FR-3.4** `chunk_id` SHALL be `chunks[0].get("chunk_id", -1)` for 1:1 backwards compat. Under N:1 this represents only the first chunk; consumers should prefer `u_ctx_id`.
**FR-3.5** `tokens` SHALL be `ctx_entry.get("tokens", 0)` (context-level total).
**FR-3.6** `extractions`, `chunk_signals`, `errors` SHALL be the result of `extractor.extract(text, doc_id, chunk_id, u_ctx_id)`, byte-identical in shape to v4/v5.
**FR-3.7** Provenance to recursive chunks (`source_u_chunk_ids`) SHALL NOT be denormalized onto the artifact record. Consumers needing the recursive-chunk chain SHALL join via `u_ctx_id` against `-logic-ctx.json::contexts[*].chunks[*].source_u_chunk_ids` (matching `generate-qa.py:393-394`'s pattern).

### FR-4 Required sidecar (no fallback)
**FR-4.1** The v5 silent-fallback path that derived `u_ctx_id` when `*-logic-ctx.json` was missing SHALL be removed.
**FR-4.2** Operators with no sidecar see the FR-1.2 "nothing to do" exit; the documented remedy is to run `_nemo.py --sdg-logical` first.

### FR-5 Idempotency, threading, mode-3 guard (preserved)
**FR-5.1** Idempotency: skip-write if `{doc_id}-logic-artifacts.json` exists and `--overwrite` is not set. Same as v5.
**FR-5.2** Mode-3 guard: rejects `recursive` and `logical` before any pool is created. Same as v5.
**FR-5.3** Within-doc thread pool: `ThreadPoolExecutor(max_workers=lx_cfg.chunk_concurrency, thread_name_prefix=doc_id)`. Same as v5.
**FR-5.4** Within-context thread pool: 2-worker pool inside `PavementExtractor.extract` (one for span, one for chunk). Same as v5.
**FR-5.5** Order preservation: the `artifacts` list SHALL be in the same order as `ctx_doc["contexts"]` (achieved by iterating the futures list in submission order, exactly as v5 does for chunks).

### FR-6 Documentation
**FR-6.1** The `extract_artifacts.py` module docstring SHALL be updated to say "Reads `{output_dir}/.../*-logic-ctx.json`" (was: `*-logic-chunks.json`).
**FR-6.2** The `--input_dir` argparse `help` text SHALL say "directory containing `*-logic-ctx.json`".
**FR-6.3** `docs/qa-generation.md` Step 2 section SHALL be updated to say `extract_artifacts.py` reads `*-logic-ctx.json` and require `_nemo.py --sdg-logical` as a prerequisite. The schema example SHALL show the v6 per-context record (with `u_ctx_id` first), and a one-line note SHALL explain that recursive-chunk provenance is reachable via join against `-logic-ctx.json` rather than denormalized onto the artifact.

## 4. Non-Functional Requirements

### NFR-1 Backward compatibility (file-level)
v5 `-logic-artifacts.json` files SHALL be cache-hit-skipped under v6 (FR-5.1). They are NOT byte-compatible with v6 records:
- v5 record key set: `{chunk_id, u_logic_chunk_id, u_ctx_id, tokens, extractions, chunk_signals, errors}` (7 keys).
- v6 record key set: `{u_ctx_id, u_logic_chunk_id, chunk_id, tokens, extractions, chunk_signals, errors}` (7 keys; same set, ordering puts `u_ctx_id` first).
- For 1:1 contexts, the values of overlapping keys MAY differ on `u_logic_chunk_id` (now sourced from sidecar) and `tokens` (now context-level), but in practice are byte-identical given current `_build_logical_contexts` output.

Operators force-regenerate via `--overwrite` for full v6 conformance.

### NFR-2 Determinism
`temperature=0.0` repeated runs SHOULD produce identical model outputs modulo OpenAI's internal nondeterminism (unchanged from v5). Iteration order of contexts is preserved (FR-5.5). v6 introduces NO new nondeterminism.

### NFR-3 Observability
Existing log signals from v5 (per-doc `"CHUNK"` summary, per-call `"NLP"` failures, per-extraction class-vocab drops) are preserved. New log lines:
- `f"{doc_id}: empty context {u_ctx_id}; skipping"` (NLP) — when a context has no chunks or an empty joined text.

### NFR-4 Performance envelope
For 1:1 contexts (current state), per-chunk timing and cost are unchanged from v5. No additional model calls; just a different file layout. Wall time on the fixture should match v5 within noise.

### NFR-5 Schema invariants
The doc-level wrapper, span-level entry shape, `chunk_signals` shape, and `errors` shape are byte-stable v5 → v6. The per-record shape changes per FR-3.1.

### NFR-6 Dependency isolation
v6 SHALL NOT add any `pip install`able dependency.

## 5. Interfaces

### 5.1 CLI interface
```text
python extract_artifacts.py [--cfg PATH] [--input_dir DIR] [--overwrite]
```
Unchanged from v5. The `--input_dir` help text changes (FR-6.2).

### 5.2 Python interface
- `LXConfig` — unchanged from v5.
- `PavementExtractor.extract(text, doc_id, chunk_id, u_ctx_id) -> dict` — signature and return shape unchanged from v5.
- `ChunkLevelExtractor.extract(text, doc_id, chunk_id) -> ChunkSignals` — unchanged from v5.
- `main(cfg, overwrite=False) -> None` — body changes per FR-1, FR-2, FR-3, FR-4, FR-5; surface unchanged.

### 5.3 File interface
Input: `{input_dir}/{doc_id}-logic-ctx.json` (changed from `-logic-chunks.json`).
Output: `{input_dir}/{doc_id}-logic-artifacts.json` (filename unchanged).

```
{
  doc_id,
  artifacts: [
    {
      u_ctx_id,
      u_logic_chunk_id,
      chunk_id,
      tokens,
      extractions: { <span-class>: [{artifact_id, u_artifact_id, text, description, significance, char_interval, attributes}, ...] },
      chunk_signals: { summary: {...}, topics: [...], terms: [...] } | null,
      errors: {span: str | null, chunk: str | null}
    }
  ]
}
```

### 5.4 Configuration interface
`extract_artifacts.toml`'s `[artifact_extraction]` block is unchanged from v5.

## 6. Acceptance Criteria

- **AC-1** `main()` source globs `*-logic-ctx.json`. Verifiable by `grep -E 'glob.*logic-ctx' extract_artifacts.py`.
- **AC-2** `main()` source no longer references `*-logic-chunks.json` as a primary glob.
- **AC-3** Per-record key set on cold-run output is `{u_ctx_id, u_logic_chunk_id, chunk_id, tokens, extractions, chunk_signals, errors}` exactly. `source_u_chunk_ids` SHALL NOT appear on the record.
- **AC-4** `u_ctx_id` is non-empty on every record. Provenance to recursive chunks is reachable via join: for every record, `u_ctx_id` matches an entry in the corresponding `-logic-ctx.json::contexts`, and that entry's `chunks[*].source_u_chunk_ids` carries the recursive-chunk ids.
- **AC-5** Cross-stage join: `{r.u_ctx_id for r in artifacts}` equals `{c.u_ctx_id for c in contexts}` for every doc on the cold-run output.
- **AC-6** Span-level entry shape unchanged: `{artifact_id, u_artifact_id, text, description, significance, char_interval, attributes}`. `chunk_signals` shape unchanged.
- **AC-7** Wall time within ±20% of v5 baseline on the same fixture (informational; no regression expected).
- **AC-8** Idempotency: second invocation without `--overwrite` writes nothing and creates no thread pools.
- **AC-9** Mode-3 guard rejects `recursive` and `logical` before any input file is read.
- **AC-10** Empty/missing-context skip path: a hand-crafted `-logic-ctx.json` with one empty `chunks: []` entry produces an `"NLP"` log line and that context's record is omitted.
- **AC-11** Static smoke check: `python -c "from extract_artifacts import LXConfig, PavementExtractor; PavementExtractor(LXConfig())"` instantiates cleanly.
- **AC-12** `extract_artifacts.py` module docstring and `--input_dir` argparse help reference `-logic-ctx.json`. `docs/qa-generation.md` updated.

## 7. Risks and Open Questions

### 7.1 Risks

- **R-1 Operators without `--sdg-logical` run.** Under v5 they got silent fallback; under v6 they get "nothing to do" and no output. Mitigation: clear docstring + CLI help; chained workflow already documented.
- **R-2 v5 cached files in the same dir.** Cache-hit skips them; operators force `--overwrite` to regenerate.
- **R-3 N:1 future bundling.** v6 handles it correctly by construction. Span `char_interval` offsets become joined-text-relative; downstream consumers needing per-chunk offsets must split on `"\n\n"` and back-resolve. Out of scope.
- **R-4 Bundle size > `max_char_buffer=10000`.** Today's bundles are well under; with N:1 we'd need to raise the buffer. Out of scope.
- **R-5 Empty contexts.** Filter + log prevents extractor calls on empty text.

### 7.2 Open Questions (non-blocking)

- **OQ-1 Drop the `chunk_id` field from the wrapper?** It's only meaningful for 1:1; under N:1 it's deprecated. Keeping it is cheap and aids 1:1 debugging. Defer; revisit if/when N:1 lands.
- **OQ-2 Add `n_chunks` field?** A small marker of how many chunks are bundled into a context (for N:1 observability). Defer; trivially derivable from `-logic-ctx.json` if needed.
- **OQ-3 Denormalize `source_u_chunk_ids` onto the artifact record (deferred).** Today the recursive-chunk provenance is reachable via join (artifact `u_ctx_id` → `-logic-ctx.json` context → `chunks[*].source_u_chunk_ids`). Adding it on the artifact record would duplicate normalized provenance for the convenience of single-file consumers. Revisit only if a downstream consumer specifically needs to walk to recursive chunks without reading `-logic-ctx.json`.
- **OQ-4 Update v5 SRS to flag v6 supersedes the per-chunk wrapper shape.** Add a one-line cross-reference at the v5 SRS NFR-1 section.
