# Plan: Extract-Artifacts v6 — Context-Driven Input

## Context

The v5 pipeline reads `*-logic-chunks.json` as its primary input and treats each chunk as the extraction unit. It loads `*-logic-ctx.json` as a sidecar only to look up `u_ctx_id` for ID minting; if the sidecar is missing, it falls back silently to a derived id. Today the resulting per-chunk and per-context records coincide (1:1 from `_build_logical_contexts`), but the two scripts in the pipeline disagree on what they consider the canonical iteration unit:

| Script | Primary input | Iteration unit | Text used |
|---|---|---|---|
| `extract_artifacts.py` (v5) | `*-logic-chunks.json` | per-chunk | `chunk.text` |
| `generate-qa.py` (current) | `*-logic-ctx.json` | per-context | `"\n\n".join(c.text for c in ctx.chunks)` |

This split costs us:
1. **Silent ID drift** if `_build_logical_contexts` ever stops being 1:1 (custom labels, N:1 bundling, etc.) — the v5 fallback formula `f"{doc_id}-ctx-{cid}"` agrees with the sidecar's ids only by coincidence today.
2. **Inconsistent text framing** — once N:1 bundling is real, `extract_artifacts.py` would extract from individual chunks while `generate-qa.py` would generate questions over joined bundle texts. Span-level `char_interval`s would point into the wrong text frame relative to the QA pipeline's view.
3. **Two iteration models for one pipeline.** Operators reasoning about "how many extraction calls per doc" need two mental models depending on which stage they're in.

v6 collapses the split: `extract_artifacts.py` reads `*-logic-ctx.json` as its primary input, iterates per context, and computes the extraction text the same way `generate-qa.py` does. The 1:1 case is byte-equivalent to v5 modulo model nondeterminism; the design becomes correct under N:1 by construction.

## Recommended approach

Single architectural shift in `extract_artifacts.py`'s `main()` plus a small ergonomic change to make the sidecar required.

### Change A — Switch primary input from chunks to contexts

In `main()`:
- Replace `chunk_dir.glob("*-logic-chunks.json")` with `chunk_dir.glob("*-logic-ctx.json")`.
- Derive `doc_id` from the sidecar filename: `chunks_path.name.replace("-logic-ctx.json", "")`.
- Output filename unchanged: `{doc_id}-logic-artifacts.json`.

The current 2-step "read chunks then read ctx-sidecar" becomes a 1-step "read ctx" — single source of truth.

### Change B — Iterate contexts, mirror `generate-qa.py`'s text construction

For each `ctx_entry` in `ctx_doc["contexts"]`:
- `u_ctx_id = ctx_entry["u_ctx_id"]`
- `chunks = ctx_entry["chunks"]` (list, length ≥ 1)
- `text = "\n\n".join(c.get("text", "") for c in chunks if c.get("text")).strip()` — identical formula to `generate-qa.py:289-292` (`build_context_text`).
- `tokens = ctx_entry.get("tokens", 0)` — context-level total (sum for N:1; equal to chunk's tokens for 1:1).
- `chunk_id = chunks[0].get("chunk_id")` — kept for 1:1 backwards compat / debugging; deprecated under N:1.
- `u_logic_chunk_id = chunks[0].get("u_logic_chunk_id", "")` — read from sidecar (single source of truth, no longer derived).

### Change C — Submit per-context to the thread pool

The within-doc pool's submission becomes one future per context (was: one per chunk). For 1:1 the count is identical. For N:1 it drops (one extraction call per bundle).

```python
with ThreadPoolExecutor(
    max_workers=lx_cfg.chunk_concurrency,
    thread_name_prefix=doc_id,
) as pool:
    futures = []
    for ctx_entry in contexts:
        chunks = ctx_entry.get("chunks", [])
        if not chunks:
            continue
        text = "\n\n".join(c.get("text", "") for c in chunks if c.get("text")).strip()
        if not text:
            logger.log("NLP", f"{doc_id}: empty context {ctx_entry.get('u_ctx_id', '?')}; skipping")
            continue
        u_ctx_id = ctx_entry.get("u_ctx_id", "")
        chunk_id = chunks[0].get("chunk_id", -1)
        u_logic_chunk_id = chunks[0].get("u_logic_chunk_id", "")
        tokens = ctx_entry.get("tokens", 0)
        futures.append((
            u_ctx_id, u_logic_chunk_id, chunk_id, tokens,
            pool.submit(extractor.extract, text, doc_id, chunk_id, u_ctx_id),
        ))
    artifacts = []
    for u_ctx_id, u_logic_chunk_id, chunk_id, tokens, fut in futures:
        result = fut.result()
        artifacts.append({
            "u_ctx_id": u_ctx_id,
            "u_logic_chunk_id": u_logic_chunk_id,
            "chunk_id": chunk_id,
            "tokens": tokens,
            "extractions": result["extractions"],
            "chunk_signals": result["chunk_signals"],
            "errors": result["errors"],
        })
```

Provenance from artifact → recursive chunks is preserved by joining the artifact's `u_ctx_id` to `-logic-ctx.json::contexts[*].chunks[*].source_u_chunk_ids` — no duplication on the artifact record.

### Change D — Make `-logic-ctx.json` required (drop the fallback)

Remove the silent-fallback branch that derives `u_ctx_id` when the sidecar is absent. If the per-doc sidecar is the primary input, "missing sidecar" means the doc is just absent from the corpus — handled by the existing "no `*-logic-ctx.json` found; nothing to do" early-exit. Operators must run `_nemo.py --sdg-logical` (or chained `--chunk-only --sdg-logical`) before `extract_artifacts.py`.

### Change E — Update docstring and CLI help

- `extract_artifacts.py` module docstring: change "Reads `*-logic-chunks.json`" → "Reads `*-logic-ctx.json`".
- `--input_dir` argparse help: "directory containing `*-logic-ctx.json`".
- `docs/qa-generation.md`: update the standalone-step description to say `extract_artifacts.py` reads `-logic-ctx.json`, not `-logic-chunks.json`. Note the prerequisite: run `_nemo.py --sdg-logical` first.

### Change F — Output schema (per-context records)

The artifacts list now carries one record per context, with `u_ctx_id` as the primary key. Per-record shape:

```
{
  u_ctx_id,                    // primary identifier; matches -logic-ctx.json and generate-qa.py
  u_logic_chunk_id,            // chunks[0].u_logic_chunk_id from -logic-ctx.json
  chunk_id,                    // chunks[0].chunk_id; for 1:1 cases only — deprecated under N:1
  tokens,                      // context-level total
  extractions: { ... },        // span-level (v4 6-key entries unchanged)
  chunk_signals: { ... } | null, // ChunkSignals dump (v4 shape unchanged)
  errors: {span, chunk},       // (v4/v5 shape unchanged)
}
```

For 1:1 the record is byte-equivalent to v5's per-chunk record modulo:
- `u_logic_chunk_id` now sourced from the sidecar instead of derived (same value if `_build_logical_contexts` keeps its current convention).
- `tokens` now sourced from the context's aggregate (same value for 1:1).

Provenance to recursive chunks (`source_u_chunk_ids`) is **not** denormalized onto the artifact record. The chain artifact → `u_ctx_id` → `-logic-ctx.json::contexts[*].chunks[*].source_u_chunk_ids` is intact via join. Adding it on the artifact would duplicate normalized provenance with no concrete consumer benefit (`generate-qa.py` already reads both files).

## Critical files

- `extract_artifacts.py` — Changes A, B, C, D, E, F (all in `main()` + module docstring + CLI help).
- `docs/qa-generation.md` — update prerequisite note + the schema example to use the new per-context record.
- `plans/srs-extract-artifacts-v5-concurrency.md` — flag in §7 OQ that v6 collapsed the per-chunk vs per-context split.

Reused utilities:
- `generate-qa.py:289-292 (build_context_text)` — formula adopted verbatim.
- `generate-qa.py:380-394` — iteration pattern adopted verbatim.

## Verification

```bash
cd /Users/igor/dev/llm/pavement-gpt/nvidia-pipeline/nvidia-sdq_custom

# Run the full chain to ensure -logic-ctx.json sidecars exist
.venv/bin/python _nemo.py --chunk-only --cfg cfg/nemo.toml
.venv/bin/python _nemo.py --sdg-logical --cfg cfg/nemo.toml

# v6 cold run (replaces -logic-chunks.json input with -logic-ctx.json)
rm -f data/_test/chunk_test-random-logic2/doc-chunks_256_random_logical/*-logic-artifacts.json
time .venv/bin/python extract_artifacts.py --cfg ./extract_artifacts.toml

# Per-record shape conforms to v6 (one record per context, u_ctx_id primary)
.venv/bin/python -c "
import json, glob
for p in glob.glob('data/_test/chunk_test-random-logic2/doc-chunks_256_random_logical/*-logic-artifacts.json'):
    d = json.load(open(p))
    assert 'doc_id' in d and 'artifacts' in d
    rec_keys = {'u_ctx_id', 'u_logic_chunk_id', 'chunk_id',
                'tokens', 'extractions', 'chunk_signals', 'errors'}
    for r in d['artifacts']:
        assert set(r.keys()) == rec_keys, (p, set(r.keys()) ^ rec_keys)
        assert r['u_ctx_id'], (p, r)
        assert 'source_u_chunk_ids' not in r, (p, 'denormalized provenance leaked')
    ids = [r['u_ctx_id'] for r in d['artifacts']]
    print(f'{p}: {len(ids)} contexts; first u_ctx_id = {ids[0]}')
"

# Cross-stage join: generate-qa.py can consume the v6 artifacts file (same join key)
.venv/bin/python -c "
import json
ctx = json.load(open('data/_test/chunk_test-random-logic2/doc-chunks_256_random_logical/TBF000131_UKN000-logic-ctx.json'))
art = json.load(open('data/_test/chunk_test-random-logic2/doc-chunks_256_random_logical/TBF000131_UKN000-logic-artifacts.json'))
ctx_ids = {c['u_ctx_id'] for c in ctx['contexts']}
art_ids = {a['u_ctx_id'] for a in art['artifacts']}
assert ctx_ids == art_ids, (ctx_ids ^ art_ids)
print(f'Join OK: {len(ctx_ids)} contexts; identical u_ctx_id sets')
"

# Idempotency: re-run is sub-second
time .venv/bin/python extract_artifacts.py --cfg ./extract_artifacts.toml

# 1:1 byte-equivalence vs v5 (modulo nondeterminism)
# - Span extractions: identical inputs ⇒ identical model outputs at temp=0 (modulo OpenAI nondeterminism).
# - chunk_signals: same.
# - Per-record shape: same key set as v5 (`u_ctx_id` reordered to first); no new fields.
```

Expected: cold run wall time matches v5 (no perf delta; the change is structural, not algorithmic). Output records carry `u_ctx_id` as the primary key; `chunk_id` field still present for 1:1 backwards compat.

## Risks

1. **Operators with v5 outputs in the same dir:** v5 records have shape `{chunk_id, u_logic_chunk_id, u_ctx_id, tokens, extractions, chunk_signals, errors}`; v6 has the same set with `u_ctx_id` reordered to first. v6 idempotency cache-hits on `{doc_id}-logic-artifacts.json` regardless of internal shape, so a v5 file would skip-write under v6. To regenerate under v6 (canonical key order + sidecar-sourced field values), operators run with `--overwrite`.
2. **Operators who never ran `--sdg-logical`:** under v5 they got the silent-fallback path; under v6 they get an early "no `*-logic-ctx.json` found; nothing to do" exit. Fix: run `_nemo.py --sdg-logical` first.
3. **N:1 future bundling:** v6 handles correctly by construction (one extraction call per bundled context with joined text). Span `char_interval` offsets become joined-text-relative; consumers needing per-chunk offsets must split on `"\n\n"` and back-resolve — not v6 scope.
4. **Bundle size > `max_char_buffer=10000`:** today's bundles are well under; with N:1 we'd need to raise the buffer or accept langextract sub-chunking. Not v6 scope.
5. **Empty contexts:** filter + log prevents the extractor from being called on empty text.
