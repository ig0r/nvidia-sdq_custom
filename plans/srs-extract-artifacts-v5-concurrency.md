# Software Requirements Specification: Extract-Artifacts v5 — Threaded Concurrency

**Feature:** Add per-doc and per-chunk threaded concurrency to the v4 extraction pipeline. The v4 contract (per-chunk wrapper shape, span and chunk call semantics, `errors` dict shape, `artifact_id` format, idempotency, mode-3 guard) is preserved verbatim. Concurrency is a tuning knob; v5 outputs are byte-compatible with v4 outputs modulo model nondeterminism.
**Component:** `nvidia-sdq_custom`
**Version:** 0.5 (draft)
**Status:** Proposed
**Companion plan:** `plans/plan-extract-artifacts-v5-concurrency.md`

---

## 1. Introduction

### 1.1 Purpose
This SRS defines requirements for adding threaded concurrency to the v4 artifact extraction pipeline so that wall-time per corpus run drops by ~8-16× (at `chunk_concurrency=8`, the tier-2 default) without changing any of v4's output contracts. The pipeline is I/O-bound on inference latency; both span (langextract) and chunk (OpenAI Structured Outputs) calls are sync and thread-safe at the SDK level. v5 wraps span+chunk submissions in a 2-worker pool inside `PavementExtractor.extract` and the per-doc chunk loop in an N-worker pool in `main()`, with N configurable via `LXConfig.chunk_concurrency`.

### 1.2 Scope

In scope:
- New `LXConfig.chunk_concurrency: int = 8`. New TOML key `chunk_concurrency` in `[artifact_extraction]`.
- Refactor `PavementExtractor.extract` to submit span and chunk calls to a 2-worker `concurrent.futures.ThreadPoolExecutor`. Per-call failure isolation preserved: each `future.result()` is wrapped in its own `try`/`except` that populates the corresponding `errors` slot.
- Refactor `main()`'s per-doc inner chunk loop to submit all chunks to a `chunk_concurrency`-worker `ThreadPoolExecutor` and iterate the futures list in submission order so per-doc records are appended in `chunk_id` order (preserving v4 FR-6.2).
- Bump `OpenAI` client `max_retries` from default 2 to 5 in `ChunkLevelExtractor.__init__` to absorb burst 429s under concurrency.
- SRS NFR addition: log lines from concurrent chunks may interleave but each carries `doc_id` and `chunk_id` and remains parseable.

Out of scope:
- v4 contract changes. Per-chunk wrapper shape (`{chunk_id, tokens, extractions, chunk_signals, errors}`), per-extraction shape, `artifact_id` format, mode-3 guard, idempotency, prompt files, Pydantic models — all unchanged.
- Reducing `extraction_passes` 2 → 1 (quality risk; deferred).
- Async refactor (rejected; rationale in `plans/plan-extract-artifacts-v5-concurrency.md` *Threading vs async — evaluation* section).
- OpenAI Batch API (24 h latency).
- langextract multi-doc batching via `text_or_documents=[list]` (more invasive; revisit if ThreadPool proves insufficient).
- Cross-doc parallelism (within-doc gives the bulk of the win; within-doc is already cleanly per-output-file).
- Explicit prompt-cache control (OpenAI auto-caches stable prefixes ≥1024 tokens; v4's span prefix qualifies; benefit accrues without code changes).

### 1.3 Definitions
- **v4** — prior state. Two sequential calls per chunk (span via langextract, chunk via OpenAI Structured Outputs). Sequential per-doc chunk loop.
- **v5** — state introduced by this SRS. Per-chunk: span and chunk run in 2 threads. Per-doc: N chunks run in `chunk_concurrency` threads.
- **Within-chunk pool** — fixed-size 2-worker `ThreadPoolExecutor` inside `PavementExtractor.extract`. Hardcoded at 2; not operator-tunable (only two calls exist).
- **Within-doc pool** — `chunk_concurrency`-worker `ThreadPoolExecutor` in `main()`'s per-doc loop. Operator-tunable via TOML.

### 1.4 References
- v4 plan + SRS: `plans/plan-extract-artifacts-v4-chunk-signals.md`, `plans/srs-extract-artifacts-v4-chunk-signals.md`.
- v5 plan: `plans/plan-extract-artifacts-v5-concurrency.md`.
- `concurrent.futures.ThreadPoolExecutor` (Python stdlib).
- OpenAI Python SDK `max_retries` parameter (`openai==1.91.0`).
- Loguru thread-safety guarantees.

---

## 2. Overall Description

### 2.1 Product Perspective
v4 is correct but slow on the fixture: ~3-5 s/chunk × 46 chunks ≈ 2-4 minutes. Both calls per chunk are independent, and chunks within a doc are independent — but all run sequentially. v5 introduces threaded concurrency at two levels (within-chunk for span vs chunk; within-doc for chunks) without touching the v4 output contract. The doc-level wrapper, per-chunk wrapper, per-extraction shapes, and chunk_signals shape are preserved byte-for-byte.

### 2.2 User Classes
- **Pipeline operator** — runs `python extract_artifacts.py`. Sees corpus runs complete much faster. May tune `chunk_concurrency` based on their OpenAI rate-limit tier.
- **Pipeline developer** — extends or tunes the extractor. Reads thread-safety constraints when adding new shared state.
- **Downstream consumer** — unchanged from v4; reads the same per-chunk shape.

### 2.3 Operating Environment
Identical to v4: Python 3.11+ (venv at 3.12), `langextract==1.2.1`, `openai==1.91.0`, `pydantic==2.11.7`, `loguru`, `python-dotenv`, `OPENAI_API_KEY` populated. `concurrent.futures` is stdlib — no new dependency.

### 2.4 Constraints
- The v4 per-chunk wrapper shape (`{chunk_id, tokens, extractions, chunk_signals, errors}`) SHALL NOT change.
- The v4 per-extraction shape (span-level: 6 keys; chunk-level: structured Pydantic shape inside `chunk_signals`) SHALL NOT change.
- The `errors` dict SHALL retain its v4 shape: `{"span": str|null, "chunk": str|null}`.
- Within each per-doc artifacts list, records SHALL appear in ascending `chunk_id` order.
- Within each chunk, the span `ext_idx` counter SHALL start at 0 and increment per emitted span-level extraction (unchanged from v4).
- `chunk_concurrency` SHALL be a positive integer. `chunk_concurrency = 1` SHALL produce output byte-equivalent to the v4 sequential implementation modulo model nondeterminism.
- The within-chunk pool size SHALL be exactly 2 (not operator-tunable).
- The `OpenAI` client used for the chunk call SHALL be instantiated once per `ChunkLevelExtractor` and shared across threads. `langextract` SHALL continue to instantiate its own httpx client per `lx.extract` call (existing behavior).

### 2.5 Assumptions
- `langextract.extract()` is thread-safe when called from multiple threads with independent `cfg` and per-call inputs (langextract instantiates its own client per call).
- The `OpenAI` SDK is thread-safe across method calls on the same `OpenAI()` instance (documented).
- Loguru handlers are thread-safe (documented).
- The fixture is correctly configured (`extract_artifacts.toml` `[paths].input_dir` points at a real directory).

---

## 3. Functional Requirements

### FR-1 LXConfig and TOML
**FR-1.1** `LXConfig` SHALL gain `chunk_concurrency: int = 8` (alongside `extraction_passes`).
**FR-1.2** The `[artifact_extraction]` block in `extract_artifacts.toml` SHALL set `chunk_concurrency = 8` by default. Comment SHALL note: "1 = sequential (v4 baseline); 8 = default (tier 2); raise to 16 on tier 3+."
**FR-1.3** `chunk_concurrency` values < 1 SHALL not be supported; the dataclass default ensures correctness when the TOML key is omitted.

### FR-2 Within-chunk parallelism
**FR-2.1** `PavementExtractor.extract` SHALL submit the span call (`self._extract_spans(...)`) and the chunk call (`self.chunk_extractor.extract(...).model_dump()`) to a `concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"chunk{chunk_id}")`.
**FR-2.2** Each future's `.result()` SHALL be awaited in a dedicated `try`/`except`. A span-task exception SHALL log `"NLP"`, set `errors["span"] = str(exc)`, and leave `extractions = {}`. A chunk-task exception SHALL log `"NLP"`, set `errors["chunk"] = str(exc)`, and leave `chunk_signals = None`.
**FR-2.3** The function SHALL return the v4 dict `{"extractions": ..., "chunk_signals": ..., "errors": ...}` unchanged in shape.

### FR-3 Within-doc parallelism
**FR-3.1** `main()`'s per-doc inner loop SHALL replace the sequential `for chunk in texts:` block with submission of all chunks to a `concurrent.futures.ThreadPoolExecutor(max_workers=lx_cfg.chunk_concurrency, thread_name_prefix=f"{doc_id}")`.
**FR-3.2** The submission SHALL preserve per-chunk metadata (`chunk_id`, `tokens`) alongside each future for use when assembling the artifacts list.
**FR-3.3** The artifacts list SHALL be assembled by iterating the futures list in submission order (not `as_completed`), calling `future.result()` per future, and appending one record per chunk in `chunk_id` ascending order.
**FR-3.4** A `future.result()` exception SHALL propagate to `main()` (the v4 behavior already wraps each chunk's call in try/except inside `PavementExtractor.extract`; v5 preserves that). If `PavementExtractor.extract` itself raises (which it should not under normal operation), the exception propagates out of `main()` and aborts the corpus run — matching v4 behavior.

### FR-4 Retry budget
**FR-4.1** `ChunkLevelExtractor.__init__` SHALL instantiate the `OpenAI` client with `max_retries=5` (up from default 2) to absorb burst 429s under concurrency.
**FR-4.2** No retry budget change SHALL be applied to the langextract span call (langextract uses its own httpx client; we don't expose its retry config). Span 429s SHALL surface as `errors["span"]` strings.

### FR-5 Output contract preservation
**FR-5.1** The doc-level shape (`{doc_id, artifacts: [...]}`) SHALL NOT change.
**FR-5.2** The per-chunk wrapper shape SHALL be `{chunk_id, tokens, extractions, chunk_signals, errors}` — identical to v4.
**FR-5.3** Per-chunk records SHALL appear in ascending `chunk_id` order within each doc's `artifacts` list (preserved by FR-3.3).
**FR-5.4** Span-level entries inside `extractions` SHALL retain the v4 6-key shape (`{artifact_id, text, description, significance, char_interval, attributes}`); chunk_signals SHALL retain the v4 Pydantic-validated shape.
**FR-5.5** `errors` SHALL be a dict with exactly two keys (`"span"`, `"chunk"`). Each value SHALL be `null` on success or a string on failure. Both keys SHALL always be present.

### FR-6 Idempotency
**FR-6.1** The doc-level cache-hit gate (skip-write when `{doc_id}-logic-artifacts.json` exists and `--overwrite` is not set) SHALL run BEFORE any thread pool is created. No thread is spawned for a cached doc.
**FR-6.2** Mode-3 guard at `_nemo.py:439` and inside `main()` SHALL run BEFORE any thread pool is created.

---

## 4. Non-Functional Requirements

### NFR-1 Backward compatibility
v4 `-logic-artifacts.json` files SHALL remain valid v5 outputs (shape unchanged). The script SHALL skip-write existing v4 outputs under v5 idempotency. Operators force-regenerate via `--overwrite` only if they want fresh runs.

**Note (v6 supersedes):** v6 (`plans/srs-extract-artifacts-v6-context-driven.md`) reorders the per-record key set to put `u_ctx_id` first and switches the primary input from `*-logic-chunks.json` to `*-logic-ctx.json`. v5's per-chunk wrapper shape is preserved at the field-set level but the iteration unit and primary key change.

### NFR-2 Determinism
With `temperature=0.0`, repeated runs SHOULD produce identical output modulo OpenAI's internal nondeterminism (unchanged from v4). Chunk-order within each doc is preserved (FR-5.3). Span `ext_idx` numbering is per-chunk-local (unchanged). v5 introduces NO new nondeterminism beyond what v4 already accepts.

### NFR-3 Observability
`"CHUNK"`-level summary/cache-hit lines (per-doc): unchanged from v4. `"NLP"` log lines from concurrent chunks MAY interleave; each line continues to carry `doc_id` and `chunk_id` so log post-processing remains parseable. Operators relying on monotonic per-chunk log ordering should sort by these fields.

### NFR-4 Performance envelope
- Per-chunk wall time: ~3-5 s sequential (v4) → ~2-3 s with within-chunk parallelism alone (Change A).
- Per-corpus wall time on the 46-chunk fixture: ~2-4 minutes (v4) → ~15-30 s at `chunk_concurrency=8` (8-16× improvement).
- Per-chunk cost: unchanged (~$0.0024 at gpt-4o-mini input prices). v5 reduces wall time, not per-chunk model spend.
- Memory overhead: ~8 MB stack × N threads ≈ 64 MB at `chunk_concurrency=8`. Trivial.

### NFR-5 Thread safety
- The `OpenAI` client is shared across threads in `ChunkLevelExtractor` (one instance per process; documented thread-safe).
- `langextract.extract()` instantiates its own client per call (one per thread) — per-call isolation, no shared state.
- Loguru is documented thread-safe.
- All v4 module-level constants (`SPAN_LEVEL_CLASSES`, `SPAN_LEVEL_EXAMPLES`, the Pydantic model classes, the `Literal[...]` aliases) are immutable and SHALL NOT be mutated by either thread.

### NFR-6 Dependency isolation
v5 SHALL NOT add any `pip install`able dependency. `concurrent.futures.ThreadPoolExecutor` is Python stdlib.

---

## 5. Interfaces

### 5.1 CLI interface
```text
python extract_artifacts.py [--cfg PATH] [--input_dir DIR] [--overwrite]
```
**Unchanged from v4.**

### 5.2 Python interface
```python
@dataclass
class LXConfig:
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    temperature: float = 0.0
    extraction_passes: int = 2
    chunk_concurrency: int = 8                              # NEW (v5)
    max_char_buffer: int = 10000
    prompt_name: str = "nemo_logic-artifacts-04-span"
    chunk_prompt_name: str = "nemo_logic-artifacts-04-chunk"
    prompt_lib: str = "./prompts"

class PavementExtractor:
    def __init__(self, cfg: LXConfig) -> None: ...           # surface unchanged
    def extract(self, text: str, doc_id: str, chunk_id: int) -> dict: ...
        # body now wraps span + chunk calls in ThreadPoolExecutor(max_workers=2)
        # returns the same v4 dict: {"extractions", "chunk_signals", "errors"}

class ChunkLevelExtractor:
    def __init__(self, cfg: LXConfig) -> None: ...           # OpenAI client now uses max_retries=5
    def extract(self, text: str, doc_id: str, chunk_id: int) -> ChunkSignals: ...

def main(cfg: dict, overwrite: bool = False) -> None: ...    # per-doc loop now uses ThreadPoolExecutor(max_workers=lx_cfg.chunk_concurrency)
```

### 5.3 Configuration interface
The `[artifact_extraction]` block SHALL gain one key:
```toml
chunk_concurrency = 8   # 1 = sequential (v4 baseline); 8 = default (tier 2); raise to 16 on tier 3+
```
All other keys are unchanged.

### 5.4 File interface
`-logic-artifacts.json` shape: byte-stable from v4. v5 produces structurally identical output modulo model nondeterminism. See `plans/srs-extract-artifacts-v4-chunk-signals.md` §5.3 for the full schema.

### 5.5 Environment interface
- `OPENAI_API_KEY` — required (unchanged from v4).

---

## 6. Acceptance Criteria

- **AC-1** `LXConfig()` exposes `chunk_concurrency` field with default value `8`. All other v4 defaults preserved.
- **AC-2** `extract_artifacts.toml`'s `[artifact_extraction]` block contains `chunk_concurrency = 8` (verifiable via `grep -E '^\s*chunk_concurrency\s*=' extract_artifacts.toml`).
- **AC-3** `PavementExtractor.extract` source contains `ThreadPoolExecutor(max_workers=2, ...)` and submits both span and chunk tasks to it.
- **AC-4** `main()` source contains `ThreadPoolExecutor(max_workers=lx_cfg.chunk_concurrency, ...)` and iterates the futures list in submission order.
- **AC-5** `ChunkLevelExtractor.__init__` instantiates `OpenAI(api_key=..., max_retries=5)`.
- **AC-6** `errors` dict shape unchanged: every per-chunk record has `errors` as a dict with exactly the keys `"span"` and `"chunk"`. Verifiable on cold-run output.
- **AC-7** Per-chunk wrapper shape unchanged: `{chunk_id, tokens, extractions, chunk_signals, errors}` on every record.
- **AC-8** Within each doc's `artifacts` list, `chunk_id` values are strictly ascending (preserved order under threading).
- **AC-9** Cold run on the fixture under v5 with `chunk_concurrency=8` completes in ≤ 30 s (target; informational, not strictly blocking — depends on network and rate limit).
- **AC-10** Setting `chunk_concurrency = 1` in TOML produces output structurally equivalent to v4 sequential output.
- **AC-11** Mode-3 guard rejects `recursive` and `logical` BEFORE any thread pool is constructed.
- **AC-12** Idempotency: a second invocation without `--overwrite` writes nothing and creates no thread pools (sub-second on the fixture).
- **AC-13** No 429 errors surface in `errors.span` or `errors.chunk` on a clean cold run with `chunk_concurrency=8` (informational; if hit, drop concurrency).
- **AC-14** Static smoke check: `python -c "from extract_artifacts import LXConfig, PavementExtractor; ..."` instantiates cleanly with v5 config; both prompts load.

---

## 7. Risks and Open Questions

### 7.1 Risks

- **R-1 Burst 429s under concurrency.** At `chunk_concurrency=8` with 3 calls/chunk, up to 24 model calls may be in-flight per doc. Tier 2 OpenAI rate limits (5 K RPM, 2 M TPM for gpt-4o-mini) accommodate sustained ~5 RPS comfortably. Bursty completion timing can push transient peaks. Mitigation: `max_retries=5` on the OpenAI client (FR-4.1). If still hit, drop the TOML default to `chunk_concurrency = 4`.
- **R-2 Log interleaving.** `"NLP"` lines from concurrent chunks arrive interleaved. Each line carries `doc_id` and `chunk_id`, so machine post-processing is unaffected; humans tailing the log see non-monotonic chunk progress. Documented in NFR-3.
- **R-3 langextract httpx-client overhead.** `langextract` instantiates a new client per call (one per thread); at `chunk_concurrency=16+` the per-thread instantiation cost may become measurable. At `8` it's negligible.
- **R-4 Connection pool exhaustion.** Default OpenAI httpx pool is 100 connections; we use ≤16 concurrent. No risk.
- **R-5 Memory.** ~8 MB/thread × ≤16 threads = ≤128 MB. Trivial.
- **R-6 Determinism under threads.** Output ordering is preserved by FR-3.3. No new model-level nondeterminism is introduced.
- **R-7 Failure isolation regression.** The within-chunk pool's per-future try/except is the only place the contract could break. The implementation's per-try-block scope (FR-2.2) is identical to v4's per-call try/except scope.

### 7.2 Open Questions (non-blocking)

- **OQ-1 Tune `chunk_concurrency` upward on tier 3+.** Tier 3 operators with 10 K+ RPM headroom can run `chunk_concurrency = 16` or higher for further wall-time reduction. No code change needed.
- **OQ-2 Switch to async if langextract gets an async API.** Currently `langextract.extract()` is sync-only. If langextract publishes an async API, revisiting async (with `AsyncOpenAI` for the chunk call) becomes attractive. Plan section *Threading vs async — evaluation* covers the rationale.
- **OQ-3 Cross-doc parallelism.** v5 parallelizes within-doc only. For corpora dominated by many small docs (1-2 chunks each), within-doc parallelism gives little benefit. A doc-level outer pool could be added; defer.
- **OQ-4 Verify prompt-cache hits.** Add a debug log of `completion.usage.prompt_tokens_details.cached_tokens` in the chunk call to confirm OpenAI's auto-cache is firing on the v4 prefix. Defer.
- **OQ-5 Make within-chunk pool size configurable.** Currently hardcoded at 2. No reason to expose; revisit only if a third extraction call is added.
