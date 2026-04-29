# Plan: Extract-Artifacts v5 — Threaded Concurrency for Span and Chunk Calls

## Context

The v4 extraction pipeline (`extract_artifacts.py`) makes two model calls per chunk: a langextract span call (`extraction_passes=2`, sync) followed by a direct OpenAI Structured Outputs chunk call (sync). The corpus loop processes docs sequentially and chunks within each doc sequentially. On the 46-chunk fixture (`data/_test/chunk_test-random-logic2/doc-chunks_256_random_logical/`) wall time is ~3-5 s per chunk → ~2-4 minutes per cold run. Cost is negligible (~$0.02/run); the felt pain is wall time.

The pipeline is I/O-bound on inference latency. Both calls are independent (different SDK paths, no shared mutable state), and chunks within a doc are independent. Two parallelism wins are available:

1. **Within-chunk**: span and chunk calls run concurrently in two threads → ~40-50% per-chunk reduction.
2. **Within-doc**: chunks processed concurrently in N threads → ~N× reduction.

Combined target on the fixture: ~2-4 min → ~15-30 s (8-16× improvement at `chunk_concurrency=8`, the tier-2 default).

OpenAI auto-caches stable prefixes ≥1024 tokens; v4's span prefix (~2146 tokens) qualifies, so cache hits accrue automatically — no code change needed for that benefit.

Out of scope: `extraction_passes 2 → 1` (quality risk), the OpenAI Batch API (24 h latency), and langextract multi-doc batching via `text_or_documents=[...]` (more invasive; revisit if ThreadPool proves insufficient). Async is evaluated below and rejected for v4/v5.

## Threading vs async — evaluation

Both can deliver the target speedup. Async is genuinely viable; the question is whether it's worth the structural cost.

**Where async would help.**
- High concurrency (100s of in-flight requests) — async scales to thousands of concurrent tasks per process; threads top out at tens of threads before memory and context-switching costs bite.
- Pure-async I/O paths — when every external call has a native `await` surface.
- Cancellation/streaming — async makes these clean.

**Why async doesn't materially help our pipeline.**

1. **The dominant bottleneck (langextract span call) is sync-only.** langextract has no async API; it uses `httpx.Client` (sync) internally. To run it under asyncio we'd wrap it in `asyncio.to_thread(...)` or `loop.run_in_executor(...)`, which re-introduces a thread per call. Async-ifying the span path doesn't actually eliminate threads; it just hides them behind an event loop. At our concurrency target (4-16 chunks), the visible cost is identical.
2. **The chunk call is only ~15-25% of per-chunk wall time.** Span dominates (~2-4 s of the ~3-5 s per-chunk total). Async-ifying *just* the chunk call via `AsyncOpenAI` saves at most ~0.5 s per chunk *if it was on the critical path* — but with within-chunk threading (Change A) the chunk call already runs concurrent with span, so its latency is fully hidden.
3. **Concurrency target.** Threads at `chunk_concurrency=4-16` have ~32-128 MB total memory overhead (8 MB stack × N threads). At 100+ concurrency this would matter; at our regime it doesn't.
4. **Plumbing cost.** Async pulls through the call stack. `main()`, `PavementExtractor.extract`, `ChunkLevelExtractor.extract` all become `async def`; we add `asyncio.run(main(...))` at entry; mixing `lx.extract` (sync) with `await client.beta.chat.completions.parse(...)` (async) is awkward. None of this is hard, but it's all friction without payoff at our scale.

**Net.** Async would be the right design if (a) langextract had an async surface, (b) we were processing 1000+ concurrent chunks, or (c) we wanted streaming/cancellation features. None apply to v5. Threading at `chunk_concurrency=8` reaches the wall-time target with less code change and a smaller mental tax.

**When to revisit.** Switch to async if any of: langextract gets an async API; corpus size grows to 10K+ chunks per run; we add a real-time UI that wants to display partial results; or we measure thread overhead (memory or context-switching) becoming material.

## Recommended approach

Two structural changes in `extract_artifacts.py` plus a config knob.

### Change A — `PavementExtractor.extract`: parallelize span and chunk

Wrap span and chunk submissions in a 2-worker pool. Per-call failure isolation is preserved (each `f.result()` raises only that worker's exception, caught by its own `try`/`except`).

```python
from concurrent.futures import ThreadPoolExecutor

def extract(self, text: str, doc_id: str, chunk_id: int) -> dict:
    extractions: dict[str, list[dict]] = {}
    chunk_signals: dict | None = None
    errors: dict[str, str | None] = {"span": None, "chunk": None}

    def _span_task() -> dict[str, list[dict]]:
        return self._extract_spans(text, doc_id, chunk_id)

    def _chunk_task() -> dict | None:
        signals = self.chunk_extractor.extract(text, doc_id, chunk_id)
        return signals.model_dump()

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"chunk{chunk_id}") as pool:
        f_span = pool.submit(_span_task)
        f_chunk = pool.submit(_chunk_task)
        try:
            extractions = f_span.result()
        except Exception as exc:
            logger.log("NLP", f"{doc_id} chunk {chunk_id}: span extraction failed: {exc}")
            errors["span"] = str(exc)
        try:
            chunk_signals = f_chunk.result()
        except Exception as exc:
            logger.log("NLP", f"{doc_id} chunk {chunk_id}: chunk-signals extraction failed: {exc}")
            errors["chunk"] = str(exc)

    return {"extractions": extractions, "chunk_signals": chunk_signals, "errors": errors}
```

The within-chunk pool size of 2 is hardcoded — there are exactly two calls and no operator reason to tune.

### Change B — `main()` per-doc loop: parallelize chunks (order-preserving)

Submit all chunks at once, then iterate the futures list in submission order so the per-doc artifact records are written in `chunk_id` order (preserves FR-6.2 record sequence).

```python
from concurrent.futures import ThreadPoolExecutor

# Inside the per-doc loop, after texts = chunks_doc.get("texts", []):
with ThreadPoolExecutor(
    max_workers=lx_cfg.chunk_concurrency,
    thread_name_prefix=f"{doc_id}",
) as pool:
    futures = [
        (chunk["chunk_id"], chunk.get("tokens", 0),
         pool.submit(extractor.extract, chunk["text"], doc_id, chunk["chunk_id"]))
        for chunk in texts
    ]
    artifacts: list[dict] = []
    for chunk_id, tokens, fut in futures:
        result = fut.result()
        artifacts.append({
            "chunk_id": chunk_id,
            "tokens": tokens,
            "extractions": result["extractions"],
            "chunk_signals": result["chunk_signals"],
            "errors": result["errors"],
        })
```

### Change C — `LXConfig` + `extract_artifacts.toml`

Add one tunable to `LXConfig` (alongside `extraction_passes`):

```python
chunk_concurrency: int = 8   # per-doc thread pool size for chunk extraction (tier-2 default)
```

TOML key in `[artifact_extraction]`:

```toml
chunk_concurrency = 8   # 1 = sequential (v4 baseline); 8 = default (OpenAI tier 2); raise to 16 on tier 3+
```

### Change D — `OpenAI()` retry budget

Bump retries to absorb burst 429s under concurrency. In `ChunkLevelExtractor.__init__`:

```python
self.client: OpenAI = OpenAI(api_key=api_key, max_retries=5)
```

Default is 2; under `chunk_concurrency=8` × 3 calls/chunk = up to 24 in-flight requests, so a slightly larger retry budget mitigates bursty rate-limit hits without code changes.

## Critical files

- `extract_artifacts.py` — Changes A, B, D + add `chunk_concurrency` field to `LXConfig`.
- `extract_artifacts.toml` — add `chunk_concurrency = 8` to `[artifact_extraction]`.
- `plans/srs-extract-artifacts-v4-chunk-signals.md` — flag OQ-4 (parallelize calls) as implemented in v5; cross-reference v5 SRS.

Reused utilities: `concurrent.futures.ThreadPoolExecutor` (stdlib). No new dependencies. The existing `_extract_spans` private method, `ChunkLevelExtractor.extract`, `errors` dict contract, and `artifact_id` ordering are preserved.

## Verification

```bash
cd /Users/igor/dev/llm/pavement-gpt/nvidia-pipeline/nvidia-sdq_custom

# Baseline timing (current sequential code, before edits) — clear cache first
rm -f data/_test/chunk_test-random-logic2/doc-chunks_256_random_logical/*-logic-artifacts.json
time .venv/bin/python extract_artifacts.py --cfg ./extract_artifacts.toml

# After implementing changes A-D
rm -f data/_test/chunk_test-random-logic2/doc-chunks_256_random_logical/*-logic-artifacts.json
time .venv/bin/python extract_artifacts.py --cfg ./extract_artifacts.toml

# Sanity: chunk_id order preserved per doc
.venv/bin/python -c "
import json, glob
for p in glob.glob('data/_test/chunk_test-random-logic2/doc-chunks_256_random_logical/*-logic-artifacts.json'):
    d = json.load(open(p))
    ids = [a['chunk_id'] for a in d['artifacts']]
    assert ids == sorted(ids), (p, ids)
    keys = set(d['artifacts'][0].keys())
    assert keys == {'chunk_id', 'tokens', 'extractions', 'chunk_signals', 'errors'}, keys
    print(f'{p}: {len(ids)} chunks, order OK, shape OK')
"

# Rate-limit headroom (informational; should be empty)
.venv/bin/python extract_artifacts.py --cfg ./extract_artifacts.toml --overwrite 2>&1 | grep -iE '429|rate.?limit' || echo "no 429s"
```

Expected: ~2-4 min cold-run → ~15-30 s with `chunk_concurrency=8`. If improvement is <4×, suspect rate limiting (look for 429 hits in `errors.span` / `errors.chunk`) before threading overhead.

## Risks

1. **Burst 429s at `chunk_concurrency=8`** on the 42-chunk doc. Surfaces as `errors["span"]` / `errors["chunk"]` containing `429` strings. Mitigation: `max_retries=5` (Change D). If still hit, drop the TOML default to 4 or 2.
2. **Log interleaving.** `loguru` is thread-safe but `"NLP"` lines from concurrent chunks will arrive interleaved. Each line still carries `doc_id chunk_id` so it's parseable; documented in the SRS NFR-3.
3. **Connection pool.** `OpenAI` client is shared across threads via `ChunkLevelExtractor` (already singleton per process). `langextract` instantiates its own httpx client per call — moderate overhead but not a correctness concern.
4. **Determinism.** Chunk-order is preserved by iterating `futures` in submission order. Span `ext_idx` numbering is per-chunk-local. No new nondeterminism.
