# Plan: Chunk Relevance Filter (mode 3 / `random_logical`)

## Context

Mode 3 (`[chunking].method = "random_logical"`, implemented by `aisa/parse/chunkers.py::HybridLogicalChunker`) currently produces logical chunks that mix domain content with document boilerplate: sponsor lists, PI/contact blocks, mission statements, FHWA-style disclaimers, references/bibliography, distribution notices, and quality-assurance statements.

Concrete failure modes observed on the test corpus (`data/_test/chunk_test-random-logic/doc-chunks_256_random_logical/`):

- **TBF000011** (6 recursive pieces): piece 0 is wholly head-of-doc boilerplate (sponsors, PI bios, contact info, mission, disclaimer) — yet the logical grouper rolls it into a chunk with the title and abstract.
- **TBF000131** (84 recursive pieces, 42 logical chunks): pieces 78–80 are pure references; pieces 81–82 are FHWA contact + KEY WORDS + NOTICE; piece 83 is a Quality Assurance statement. Logical chunks 39, 40, 41 are entirely tail noise. Logical chunk 0 mixes the SUMMARY AND DISCLAIMERS section with the introduction.

This noise contaminates downstream stages: the artifact-extraction prompt produces useless metadata extractions (sponsor lists, contact info), QA generation produces noise-derived questions ("What is the FHWA's contact info?"), and Stage 1 hard-negative mining is forced to embed and rank irrelevant passages.

The fix is to score each recursive piece for pavement-engineering relevance, then run logical grouping only over the survivors. Filtering at the recursive level (before logical grouping) is preferred over filtering at the logical level because:

1. Recursive pieces are smaller and contiguous — boilerplate lands in its own piece (cleanly droppable) instead of being merged with content.
2. Logical chunks already mix boilerplate + content (TBF000131 logical chunk 0). Filtering after grouping forces an all-or-nothing decision on a mixed unit.
3. Removing noise before logical grouping lets the grouping LLM make better boundary decisions on the remaining content.
4. Cost is comparable: ~9 recursive pieces vs. ~7 logical chunks per doc.

The filter is opt-in (config flag, default off) and only active for mode 3.

## Scope

**In scope**

- New prompt file at `prompts/nemo_eval-02.txt` — closed-list noise rubric + 0/0.5/1 score schema + 6 worked examples drawn from the test corpus + field semantics for `RelevanceJudgment` + chain-of-thought instructions (`<scratchpad>` reasoning before `<json>` output). The prompt is a `.format()`-style template with a single `{CHUNK}` placeholder; the chunk's text is substituted into the placeholder for each per-piece call.
- New `QAGenerator.evaluate_chunks` async method in `_nemo.py` — fans out **one OpenAI chat completion call per recursive piece** via `await asyncio.gather(...)` over `await self.eval_client.chat.completions.create(...)` calls, bounded by `asyncio.Semaphore(self.relevance_concurrency)`. Each response is plain text containing `<scratchpad>...</scratchpad>` and `<json>...</json>` blocks; the `<json>` content is extracted by regex and validated via `RelevanceJudgment.model_validate_json()`. Per-piece exception fallback to `score=1.0`; writes `-relevance.json`; honors `self.overwrite`.
- New Pydantic v2 model in `_nemo.py`: `RelevanceJudgment` (`score: Literal[0, 0.5, 1]`, `reason: str`). Score Literal mixes int and float to match the prompt's exact wording (`"0"`, `"0.5"`, `"1"`); Pydantic accepts both `0`/`0.0` and `1`/`1.0` via `==` equality. Used for *post-receipt* validation (`model_validate_json` on the extracted `<json>` block), not API-boundary enforcement.
- New `self.eval_client: AsyncOpenAI | None` instantiated in `QAGenerator.__init__` when the filter is on. `OPENAI_API_KEY` required at construction. New `self.relevance_concurrency: int` read from `[chunking].relevance_concurrency` (default `8`).
- New free function `group_kept_pieces` in `aisa/parse/chunkers.py` — mask-aware version of the logical-grouping algorithm; runs `_llm_split_decisions` over contiguous runs of kept piece indices; returns chunks + `source_chunk_ids` referencing original (unfiltered) indices.
- Modify `QAGenerator.path2chunks` in `_nemo.py` — make async; insert the optional eval + mask-grouping stage inside the existing `if is_hybrid:` branch behind a config flag.
- Update existing async callers to `await path2chunks`: `run_chunk_only_pipeline`, `run_sgd_pipeline`, `run_sgd_logical_pipeline`.
- New config key `[chunking].relevance_filter` (bool, default `false`) honored only when `method == "random_logical"`.
- New per-doc output `{chunk_dir}/{doc_id}-relevance.json` (only when filter is on).
- Companion SRS `plans/srs-chunk-relevance-filter.md`.

**Out of scope**

- Mode 1 (`recursive`) and mode 2 (`logical`) — explicitly deferred. `LLMSemanticChunker` is not modified.
- Configurable threshold. v1 hard-codes the keep policy `score > 0.5` (only `1` survives). Threshold becomes a config knob in a follow-up if the strict default is too aggressive.
- Configurable eval model and temperature. v1 hardcodes `gpt-4o-mini` at `temperature=0.0` for the eval call (matching `extract_artifacts.py` v4). Exposing `[chunking].relevance_model` / `.relevance_temperature` is a follow-up.
- Provider portability for the eval call. The eval call is OpenAI-only in v1 because we use the OpenAI Python SDK directly (`AsyncOpenAI().chat.completions.create()`). Routing through `BaseLLM` (e.g. via `langchain.with_structured_output(RelevanceJudgment)` for providers that support it) is a follow-up.
- Bulk full-doc evaluation (all pieces tagged into one LLM call). Considered and **rejected** during prototyping — the model used document context to over-extend a "References section" classification 4–6 pieces backward into body content. v1 evaluates each recursive piece in its own call. See "Decisions flagged" for full rationale.
- Per-piece result caching (one cache file per piece). v1 keeps the per-doc `{doc_id}-relevance.json` cache shape; the file is written once after all per-piece calls complete. Per-piece caching could enable mid-doc resume after partial failure but isn't needed in v1.
- Schema-version markers on `-relevance.json`. Operators force-regenerate via `self.overwrite = True` when the prompt changes (same convention as the rest of the pipeline).
- Manual-review UI / annotator workflow. Hand-labeling for verification uses plain JSON files in `_test/chunk_test-random-logic/relevance_truth/`.
- CLI flag for the filter. v1 is TOML-only.
- Migration tooling. Existing mode-3 outputs without the filter remain valid; the new code path is purely additive.

## Concrete changes

### New: `prompts/nemo_eval-02.txt`

Plain-text prompt template with a single `{CHUNK}` placeholder, used as a `.format()`-compatible template. The chunk text is substituted into `{CHUNK}` and the resulting string is sent as a single *user* message to the chat completion. The prompt instructs the model to:

1. Reason about the chunk inside `<scratchpad>...</scratchpad>` tags (chain-of-thought).
2. Emit a JSON object inside `<json>...</json>` tags with fields `score` and `reason`.

Sections:

- **Task** — score each recursive piece for pavement-engineering relevance on the closed set `{0, 0.5, 1}`.
- **Score schema**:
  - `1` — clearly relevant pavement engineering content (concepts, methods, findings, processes, technical detail, observations, data).
  - `0.5` — unsure / mixed / ambiguous; short fragment where intent is unclear; piece mixes relevant material with closed-list noise.
  - `0` — clearly noise; matches one of the closed-list categories below.
- **Closed-list noise rubric** (label as `0` if the piece is essentially one of these):
  - Sponsor / funder lists
  - PI / author / contact blocks (names, phones, emails, addresses)
  - Mission / vision statements
  - Copyright / disclaimer / liability notices
  - Distribution & availability notices
  - Key Words / index term lists
  - References / bibliography sections (whole sections, not in-content citations)
  - Isolated figure / table captions
  - Page headers / footers / running titles
  - Table-of-contents fragments
  - Quality assurance / endorsement statements
- **In-content citations stay** — a sentence containing "(Snyder et al. 2018)" inside a content paragraph is still relevant (`1`); a section labelled "References" followed by a list of citations is `0`.
- **Field semantics** — describe the `RelevanceJudgment` fields surfaced inside `<json>`:
  - `score` — one of `0`, `0.5`, `1`; semantics defined in the score schema above.
  - `reason` — brief explanation, ≤15 words.
- **Output format** — `<scratchpad>` block (free-form reasoning) followed by `<json>` block (`{score, reason}`). One example output is shown in the prompt body.
- **Worked examples** — 6 examples drawn from the test corpus covering each score class. At minimum: a clear `0` (references list, QA statement), a clear `0.5` (mixed boilerplate + content fragment), a clear `1` (substantive technical content), a project-case-study `1` (so the model doesn't confuse case studies near the references section with the references themselves), and a boundary-chunk `0.5` (conclusions + figure captions + start of REFERENCES heading).

The prompt does NOT mention the downstream keep threshold — labels are produced honestly across the three classes; the threshold lives in code. The `<json>` shape is enforced by post-receipt Pydantic validation (`RelevanceJudgment.model_validate_json`), not by OpenAI's Structured Outputs API path. The model is allowed to think in `<scratchpad>` first; if its `<json>` is malformed or missing, the per-piece exception fallback yields `score=1.0` (preserves recall under the strict policy).

### New: Pydantic model in `_nemo.py`

Defined at module level, near the top of `_nemo.py`:

```python
from typing import Literal
from pydantic import BaseModel, Field

RELEVANCE_SCORE = Literal[0, 0.5, 1]

class RelevanceJudgment(BaseModel):
    score: RELEVANCE_SCORE = Field(
        description="1=clearly relevant pavement engineering content; 0.5=unsure/mixed; 0=closed-list noise"
    )
    reason: str = Field(description="Brief explanation, ≤15 words")
```

`RELEVANCE_SCORE` is a mixed-type `Literal` of `0` (int), `0.5` (float), `1` (int) — matching the prompt's exact wording (`"Must be exactly 0, 0.5, or 1"`). Pydantic v2 validates by `==` equality, so JSON values `0`, `0.0`, `1`, `1.0`, and `0.5` all match successfully; values like `0.7` or `2` are rejected with `ValidationError`.

The model is used for **post-receipt validation** via `RelevanceJudgment.model_validate_json(json_block)` after extracting the `<json>` content from the raw response — *not* as `response_format=RelevanceJudgment` with the Structured Outputs path. The chain-of-thought `<scratchpad>` block in the response is incompatible with Structured Outputs (which requires pure JSON output). Trade-off: schema enforcement is post-receipt rather than at the API boundary; on malformed output the per-piece exception fallback kicks in (FR-2.6 in the SRS).

There is **no wrapper / list model** (formerly `RelevanceResponse`). Each per-piece call returns one `RelevanceJudgment`; the loop in `evaluate_chunks` aggregates them. This eliminates: coverage soft-validation (every piece has its own call, so missing-`chunk_id` cases can't occur); duplicate-`chunk_id` handling; out-of-range-`chunk_id` handling. Soft-validation collapses to per-piece exception handling (covers malformed JSON, missing `<json>` tags, off-enum scores, network errors, refusals).

### New: `QAGenerator.evaluate_chunks` in `_nemo.py`

Per-piece fan-out via `asyncio.gather`, bounded by `asyncio.Semaphore`. Each chunk is substituted into the prompt's `{CHUNK}` placeholder and sent as a single user message. The response contains `<scratchpad>...</scratchpad>` reasoning followed by `<json>...</json>` containing `{score, reason}`; both blocks are extracted by regex.

```python
import re

_JSON_RE = re.compile(r"<json>\s*(.*?)\s*</json>", re.DOTALL)
_SCRATCH_RE = re.compile(r"<scratchpad>\s*(.*?)\s*</scratchpad>", re.DOTALL)
_FENCE_OPEN_RE = re.compile(r"^```\w*\s*", re.MULTILINE)
_FENCE_CLOSE_RE = re.compile(r"\s*```\s*$", re.MULTILINE)


async def evaluate_chunks(self, file_path: Path, chunks: dictlist) -> dictlist:
    """Score each recursive chunk for pavement-engineering relevance via OpenAI
    chat completions with chain-of-thought + tagged-JSON output. Each chunk gets
    its own LLM call; calls run concurrently bounded by self.relevance_concurrency.

    Caller (path2chunks) only invokes this when method == "random_logical" and
    [chunking].relevance_filter is true. The method itself does not gate on mode.
    Assumes self.eval_client is not None.
    """
    base_out: str = self.doc_paths[file_path].replace(
        "-chunks.json", "-relevance.json"
    )
    if Path(base_out).exists() and not self.overwrite:
        cached = files.read_json(base_out)
        scores: dictlist = (
            cached.get("scores", []) if isinstance(cached, dict) else cached
        )
        logger.log("CHUNK", f"{file_path.name}: cache hit -> {base_out}")
        return scores

    prompt_template: str = self.llm.read_prompt("nemo_eval-02")
    sem = asyncio.Semaphore(self.relevance_concurrency)

    async def _eval_one(chunk: dict) -> dict:
        async with sem:
            try:
                user_content = prompt_template.format(CHUNK=chunk["text"])
                completion = await self.eval_client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.0,
                    messages=[{"role": "user", "content": user_content}],
                )
                text = completion.choices[0].message.content or ""

                json_match = _JSON_RE.search(text)
                if not json_match:
                    raise RuntimeError(
                        f"no <json> block in response (head: {text[:200]!r})"
                    )
                json_text = json_match.group(1).strip()
                json_text = _FENCE_OPEN_RE.sub("", json_text)
                json_text = _FENCE_CLOSE_RE.sub("", json_text).strip()

                judgment = RelevanceJudgment.model_validate_json(json_text)

                scratch_match = _SCRATCH_RE.search(text)
                scratchpad = (
                    scratch_match.group(1).strip() if scratch_match else None
                )

                return {
                    "chunk_id": chunk["chunk_id"],
                    "score": float(judgment.score),
                    "reason": judgment.reason,
                    "scratchpad": scratchpad,
                }
            except Exception as exc:
                logger.log(
                    "CHUNK",
                    f"{file_path.name}: chunk_id={chunk['chunk_id']} eval failed "
                    f"({exc!r}); defaulting to score=1.0",
                )
                return {
                    "chunk_id": chunk["chunk_id"],
                    "score": 1.0,
                    "reason": f"error: {exc}",
                    "scratchpad": None,
                }

    scores: dictlist = await asyncio.gather(*[_eval_one(c) for c in chunks])
    doc_id: str = Path(self.doc_paths[file_path]).name.replace("-chunks.json", "")
    files.write_json({"doc_id": doc_id, "scores": scores}, base_out)
    return scores
```

**Soft-validation collapses to per-piece exception handling.** Per-piece coverage is automatic (one call per piece, returned alongside its `chunk_id`). The exception handler covers all post-receipt failure modes:

- Missing `<json>` block in the response (model didn't follow instructions).
- Malformed JSON inside `<json>` block (model emitted invalid syntax).
- Off-enum `score` value (e.g. `0.7`) — `RelevanceJudgment.model_validate_json` raises `pydantic.ValidationError`.
- Wrong types (e.g. `score` as string) — same.
- Missing required fields (`score` or `reason` absent) — same.
- Network errors, OpenAI API errors, refusal-as-empty-content.

In every case the fallback yields `{"score": 1.0, "reason": "error: <msg>", "scratchpad": None}` so the piece is kept under the strict `> 0.5` policy. Logged but not punitive.

There is **no `_reconcile_relevance` helper** — coverage gaps, duplicate `chunk_id`s, and out-of-range `chunk_id`s cannot occur by construction (each input piece gets its own call).

### Modify: `QAGenerator.__init__` in `_nemo.py`

Add eager `AsyncOpenAI` instantiation when the filter is on, plus the concurrency knob:

```python
import os
from openai import AsyncOpenAI

# … existing __init__ body up to chunk_cfg loading …

self.relevance_concurrency: int = int(
    self.chunk_cfg.get("relevance_concurrency", 8)
)
self.eval_client: AsyncOpenAI | None = None
filter_on: bool = bool(self.chunk_cfg.get("relevance_filter", False))
if filter_on and self.chunk_cfg.get("method") == "random_logical":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required when [chunking].relevance_filter = true"
        )
    self.eval_client = AsyncOpenAI(api_key=api_key)
elif filter_on:
    logger.log(
        "CHUNK",
        f"relevance_filter ignored: only honored for method='random_logical' "
        f"(got method={self.chunk_cfg.get('method')!r})",
    )
```

The mode-mismatch warning (previously a separate block) is folded in here. The client is `None` when the filter is off or the mode is wrong. `self.relevance_concurrency` is always set (default `8`) but only consumed by `evaluate_chunks` when the filter runs.

### New: `group_kept_pieces` in `aisa/parse/chunkers.py`

```python
def group_kept_pieces(
    pieces: list[str],
    kept_indices: list[int],
    llm: BaseLLM,
    prompt_template: str,
    window: int,
    stride: int,
    has_overlap: bool,
) -> tuple[list[str], list[list[int]]]:
    """Logical grouping over a masked subset of recursive pieces.

    Splits kept_indices into maximal contiguous runs, runs _llm_split_decisions
    over each run independently, and concatenates the results. Gaps between
    runs are implicit hard splits — no logical chunk crosses a dropped piece.
    Returned source_chunk_ids reference original (unfiltered) piece indices.
    """
    if not kept_indices:
        return [], []

    runs: list[list[int]] = []
    current: list[int] = [kept_indices[0]]
    for idx in kept_indices[1:]:
        if idx == current[-1] + 1:
            current.append(idx)
        else:
            runs.append(current)
            current = [idx]
    runs.append(current)

    all_chunks: list[str] = []
    all_sources: list[list[int]] = []
    for run in runs:
        sub_pieces: list[str] = [pieces[i] for i in run]
        if len(sub_pieces) == 1:
            all_chunks.append(sub_pieces[0])
            all_sources.append([run[0]])
            continue
        sub_splits: list[int] = _llm_split_decisions(
            llm, prompt_template, sub_pieces, window, stride
        )
        sub_chunks, sub_sources = _assemble_with_overlap_trim(
            sub_pieces, sub_splits, has_overlap
        )
        for ch, src in zip(sub_chunks, sub_sources):
            all_chunks.append(ch)
            all_sources.append([run[i] for i in src])

    return all_chunks, all_sources
```

Properties:

- `kept_indices == list(range(len(pieces)))` produces output equivalent to the current `HybridLogicalChunker.split` body (one run, no gaps).
- `kept_indices == []` returns empty.
- Adjacency (and therefore overlap) is preserved within each run; gaps break overlap chains naturally because `_assemble_with_overlap_trim` only trims between consecutive sub-list elements.
- `LLMSemanticChunker` is not modified.

### Modify: `QAGenerator.path2chunks` in `_nemo.py`

Make `path2chunks` async. Inside the existing `if is_hybrid:` branch, after the recursive split and before writing `-logic-chunks.json`, branch on the config flag:

```python
async def path2chunks(self, file_path: Path) -> dictlist:
    abs_path, filename = files.split_path(str(file_path))
    doc_id: str = "_".join(filename.split("_")[:2])
    base_out: str = f"{self.chunk_dir}/{doc_id}-chunks.json"
    self.doc_paths[file_path] = base_out
    method: str = self.chunk_cfg.get("method", "recursive")
    is_hybrid: bool = method == "random_logical"
    cache_path: str = (
        f"{self.chunk_dir}/{doc_id}-logic-chunks.json" if is_hybrid else base_out
    )

    if Path(cache_path).exists() and not self.overwrite:
        return files.read_json(cache_path).get("texts", [])

    raw_text = file_path.read_text(encoding="utf-8")
    tables = MD_PATTERNS["table"].findall(raw_text)
    images = MD_PATTERNS["image"].findall(raw_text)
    raw_text = MD_PATTERNS["table"].sub("", raw_text)
    raw_text = MD_PATTERNS["image"].sub("", raw_text)
    parsed_file: str = str(abs_path) + "/" + filename

    if is_hybrid:
        # 1. Recursive pre-split (always)
        rec_pieces: list[str] = self.chunker.recursive.split(raw_text)
        rec_chunks: dictlist = [
            {"text": p, "chunk_id": idx, "tokens": get_token_count(p)}
            for idx, p in enumerate(rec_pieces)
        ]
        files.write_json(
            {"doc_id": doc_id, "parsed_file": parsed_file,
             "texts": rec_chunks, "images": images, "tables": tables},
            base_out,
        )

        # 2. Optional relevance filter
        relevance_on: bool = bool(self.chunk_cfg.get("relevance_filter", False))
        if relevance_on:
            try:
                scores: dictlist = await self.evaluate_chunks(file_path, rec_chunks)
                kept_indices: list[int] = [
                    c["chunk_id"]
                    for c, s in zip(rec_chunks, scores)
                    if s["score"] > 0.5
                ]
                kept_count, dropped, half = self._summarize_scores(scores)
                logger.log(
                    "CHUNK",
                    f"{file_path.name}: {kept_count}/{len(rec_chunks)} pieces kept "
                    f"(filtered {dropped}; unsure {half})",
                )
            except Exception as exc:
                logger.log(
                    "CHUNK",
                    f"{file_path.name}: relevance filter failed ({exc}); falling back to keep-all",
                )
                kept_indices = list(range(len(rec_pieces)))
        else:
            kept_indices = list(range(len(rec_pieces)))

        # 3. Logical grouping (mask-aware iff filter changed kept_indices)
        from aisa.parse.chunkers import group_kept_pieces
        prompt_template: str = self.chunker.prompt_template
        raw_chunks, sources = group_kept_pieces(
            rec_pieces,
            kept_indices,
            self.llm,
            prompt_template,
            self.chunker.window,
            self.chunker.stride,
            self.chunker.recursive_overlap > 0,
        )

        logic_chunks: dictlist = [
            {"text": ch, "chunk_id": idx, "tokens": get_token_count(ch),
             "source_chunk_ids": sources[idx] if idx < len(sources) else []}
            for idx, ch in enumerate(raw_chunks)
        ]
        files.write_json(
            {"doc_id": doc_id, "parsed_file": parsed_file,
             "texts": logic_chunks, "images": images, "tables": tables},
            cache_path,
        )
        return logic_chunks

    # Non-hybrid path: unchanged
    raw_chunks: list[str] = self.chunker.split(raw_text)
    chunks: dictlist = [
        {"text": ch, "chunk_id": idx, "tokens": get_token_count(ch)}
        for idx, ch in enumerate(raw_chunks)
    ]
    files.write_json(
        {"doc_id": doc_id, "parsed_file": parsed_file,
         "texts": chunks, "images": images, "tables": tables},
        base_out,
    )
    return chunks
```

Notes:

- The recursive pre-split is now driven by `self.chunker.recursive.split(raw_text)` directly, so we always have explicit `rec_pieces` regardless of whether the filter runs. The previous code path used `self.chunker.split(raw_text)` and read back `last_recursive_pieces` / `last_source_indices`; that path is replaced inline (with equivalent behavior when `kept_indices` is full-coverage) so the filter and the unfiltered path share one shape.
- Cache short-circuit at the top is preserved: if `-logic-chunks.json` exists and `self.overwrite == False`, no LLM calls are made (relevance or grouping).
- Non-hybrid branches are untouched.
- All callers (`run_chunk_only_pipeline`, `run_sgd_pipeline`, `run_sgd_logical_pipeline`) update to `await self.path2chunks(file_path)`.

### Modify: `cfg/nemo.toml`

The flag is operator-set, not enabled by default in the committed TOML. Documented in the SRS under §2.4 / §5.4:

```toml
[chunking]
method = "random_logical"
chunk_size = 256
recursive_overlap = 50
hybrid_window = 8
hybrid_stride = 6
relevance_filter = true       # NEW; default false. Only honored when method == "random_logical".
relevance_concurrency = 8     # NEW; default 8. Bounds in-flight per-piece eval calls.
```

### Unchanged

- `aisa/parse/chunkers.py::HybridLogicalChunker.split` — public method preserved; behavior unchanged when called directly.
- `aisa/parse/chunkers.py::LLMSemanticChunker` — mode 2 chunker, untouched.
- `_chunks.json` and `-logic-chunks.json` schemas.
- `extract_artifacts`, `generate_qa_pairs`, `evaluate_qa_pairs`, `run_data_prep_pipeline` — downstream stages consume the same `-logic-chunks.json` shape.
- `BaseLLM` / `Embedder` / providers — no plumbing changes.
- `reqs.txt`.

## Behavior notes

- **Filter off (default)**: byte-identical output for `-chunks.json` and `-logic-chunks.json` to pre-feature behavior. Mode 1 / mode 2 produce identical output regardless of flag.
- **Filter on**: a new `-relevance.json` is written alongside the existing chunk files. `-logic-chunks.json` reflects only kept pieces; `source_chunk_ids` continues to reference original recursive piece indices, so dropped indices are simply absent.
- **Strict keep policy `> 0.5`**: only `score == 1.0` survives. Both `0.0` (clearly noise) and `0.5` (unsure) are dropped. Trade-off: lower recall on borderline pieces (mixed disclaimer + intro will be cut entirely); higher precision on what enters logical chunks. The `0.5` label is preserved as a distinct value (rather than collapsing to binary 0/1) so we get telemetry on uncertainty and can revisit the threshold in a follow-up without re-running the eval.
- **Schema enforcement is post-receipt, not at the API boundary**: `score` is constrained to `Literal[0, 0.5, 1]` via Pydantic, applied via `RelevanceJudgment.model_validate_json()` on the extracted `<json>` block. The OpenAI Structured Outputs path (`response_format=...`) is NOT used here because the prompt's chain-of-thought design (`<scratchpad>` then `<json>` tags) is incompatible with strict JSON-only output. Trade-off accepted: the model could in principle return malformed output (missing `<json>` tags, invalid JSON inside, off-enum scores). All such cases are caught by the per-piece exception handler.
- **Soft-validation default = `1.0` for per-piece exceptions only**: per-piece coverage is automatic (one call per piece). The exception handler now covers more cases than under Structured Outputs (missing `<json>` block, malformed JSON, off-enum scores, wrong types, refusals, network errors), but the fallback semantics are the same: `score=1.0` so the piece is kept under the strict `> 0.5` policy. Logged but not punitive. Coverage-gap, duplicate, and out-of-range `chunk_id` cases are eliminated by construction.
- **Scratchpad reasoning is captured in `-relevance.json`**: each entry gains a `scratchpad: str | null` field populated from `<scratchpad>...</scratchpad>` in the response. Useful for debugging classification decisions and for prompt iteration. `null` when the model didn't include a scratchpad block or on per-piece exceptions. Adds ~150 tokens/chunk to the output file (~50 KB per 84-chunk doc) — acceptable.
- **OPENAI_API_KEY required when filter is on**: even if `[llm].provider` is Google or Ollama, the relevance call uses OpenAI directly. `QAGenerator.__init__` raises `RuntimeError` if the key is missing and the filter is on (with `method == "random_logical"`).
- **Whole-doc eval-call exception handling**: an exception in `evaluate_chunks` itself (e.g. `read_prompt` fails, `Semaphore` ctor fails — exotic) does not abort `path2chunks`. `path2chunks` catches and falls back to `kept_indices = list(range(len(rec_pieces)))` (per-doc keep-all). Per-piece exceptions are caught inside `_eval_one` and don't propagate.
- **N concurrent LLM calls per document**: one call per recursive piece, bounded by `self.relevance_concurrency` (default `8`). For TBF000131 (84 pieces) at concurrency 8 → 11 batches → ~5–10 seconds wall time. Each call sees only one piece's text (~256 tokens) plus the system prompt (~1K tokens). No full-doc context.
- **Cost**: N OpenAI calls per doc on cold runs. At gpt-4o-mini input prices ≈ ~$0.015/doc for an 84-piece doc (system prompt amortizes across N calls; each call is small). ~5× the bulk approach, ~$1.50/100 docs in absolute terms. Cache hits eliminate re-computation. Hardcoded model means cost is independent of `[llm].model`.
- **Cache invalidation**: changing the eval prompt or the Pydantic schema requires deleting `-relevance.json` files (or running with `self.overwrite = True`). Same convention as the rest of the pipeline.
- **No mode-1 / mode-2 effect**: a `relevance_filter = true` set with `method != "random_logical"` is ignored at runtime; one `"CHUNK"` warning is emitted at construction; no `AsyncOpenAI` client is created so `OPENAI_API_KEY` is not required in that case.

## Verification

Test corpus: `data/_test/chunk_test-random-logic/doc-chunks_256_random_logical/` with TBF000011 (6 pieces) and TBF000131 (84 pieces). Hand-labeled truth at `data/_test/chunk_test-random-logic/relevance_truth/{doc_id}-relevance-truth.json`.

Cold runs:

```bash
# Filter on (mode 3)
python _nemo.py --chunk-only --cfg cfg/nemo.toml \
    --input_dir data/_test/chunk_test-random-logic/reports \
    --output_dir data/_test/chunk_test-random-logic
# After flipping [chunking].relevance_filter = true in TOML

# Filter off (mode 3) — control
# After flipping back to relevance_filter = false (or omitting the key)
```

AC harness:

- **AC-1** `prompts/nemo_eval-02.txt` exists; contains task description, closed-list noise rubric (≥11 categories), 0/0.5/1 score schema, 6 worked examples (covering each score class plus a project-case-study `1` and a boundary-chunk `0.5`), field semantics for `RelevanceJudgment` (`score`, `reason`), and chain-of-thought output instructions (`<scratchpad>` then `<json>` tags). The file SHALL contain exactly one `{CHUNK}` placeholder. The file SHALL NOT contain `<start_chunk_N>` tagging conventions or any other placeholder.
- **AC-2** `QAGenerator.evaluate_chunks` exists with the documented signature; fans out per-piece via `asyncio.gather` over `await self.eval_client.chat.completions.create(model="gpt-4o-mini", temperature=0.0, ...)` calls bounded by `asyncio.Semaphore(self.relevance_concurrency)`. Each per-piece call substitutes the chunk text into the prompt template's `{CHUNK}` placeholder via `.format(CHUNK=chunk["text"])` and sends the result as a single user message. Returns a list of length `len(chunks)`; each entry has int `chunk_id`, float `score ∈ {0.0, 0.5, 1.0}`, str `reason`, str-or-None `scratchpad`.
- **AC-2a** Pydantic model `RelevanceJudgment` imports cleanly from `_nemo`. `RelevanceJudgment.model_json_schema()` includes `score`'s `enum` constraint listing `[0, 0.5, 1]`. `RelevanceJudgment.model_fields["score"].annotation` is the `RELEVANCE_SCORE` Literal alias. There is no `RelevanceResponse` / `RelevanceItem` / list-wrapper class in `_nemo`. `RelevanceJudgment` is used for post-receipt validation via `model_validate_json` on the extracted `<json>` block, not as `response_format=...`.
- **AC-2b** Missing `OPENAI_API_KEY` with `relevance_filter = true` and `method == "random_logical"`: `QAGenerator.__init__` raises `RuntimeError` with a message naming `OPENAI_API_KEY`.
- **AC-2c** Missing `OPENAI_API_KEY` with `relevance_filter = false` (or `method != "random_logical"`): `QAGenerator.__init__` does not raise; `self.eval_client is None`.
- **AC-2d** `self.relevance_concurrency` defaults to `8` when `[chunking].relevance_concurrency` is not set; reads the configured int when set. The semaphore in `evaluate_chunks` honors this value.
- **AC-3** Cold run on TBF000011 with filter on writes `-relevance.json` containing 6 entries; runs 6 eval calls (one per piece, observable via OpenAI usage telemetry or a counter); produces `-logic-chunks.json` from kept pieces.
- **AC-4** Cold run on TBF000131 with filter on writes `-relevance.json` containing 84 entries; runs 84 eval calls. Pieces 78–83 (references / FHWA contact / KEY WORDS / NOTICE / QA statement) are scored `0`. Pieces 48–51 (project case studies + conclusions immediately before the `# REFERENCES` heading) are scored `1` or `0.5` (NOT `0`) — verifying that per-piece eval avoids the bulk-approach's body↔references over-extension. The resulting `-logic-chunks.json` has no logical chunk whose `source_chunk_ids` contains any of {78, 79, 80, 81, 82, 83}, and at least one logical chunk whose `source_chunk_ids` includes pieces from [48, 49, 50, 51].
- **AC-5** Idempotency: a second invocation with the same flag and `self.overwrite == False` reads from cache, makes zero LLM calls, produces byte-identical output.
- **AC-6** Filter off (default) on the test corpus produces `-chunks.json` and `-logic-chunks.json` byte-identical to the existing fixtures in `data/_test/chunk_test-random-logic/doc-chunks_256_random_logical/`. No `-relevance.json` is written.
- **AC-7** Mode 1 and mode 2 with and without `relevance_filter = true` produce byte-identical output between flag values; no `-relevance.json` is written; one `"CHUNK"` warning emitted at construction when flag is set with non-mode-3.
- **AC-8** `group_kept_pieces` smoke test: `kept_indices == list(range(N))` produces output equivalent to `HybridLogicalChunker.split` over the same pieces. With a synthetic gap (drop index 3 of 8), no returned `source_chunk_ids` crosses the gap; the output has at least one chunk whose sources ⊆ [0, 1, 2] and another whose sources ⊆ [4, 5, 6, 7].
- **AC-9** Per-piece exception fallback covers all post-receipt failure modes. Verified by stubbing the eval client to:
  - Return text with no `<json>` block → fallback `{"score": 1.0, "reason": "error: no <json> block ...", "scratchpad": None}`.
  - Return text with malformed JSON inside `<json>` → fallback (`json.JSONDecodeError` from Pydantic).
  - Return text with off-enum score (`"score": 0.7`) → fallback (`pydantic.ValidationError`).
  - Raise `RuntimeError` directly → fallback.
  In all cases: one `"CHUNK"` log line is emitted naming the doc, chunk_id, and exception; the failed piece's `score == 1.0` so it remains in `kept_indices`; other pieces complete normally. Missing-`chunk_id` coverage gaps, duplicate `chunk_id`s, and out-of-range `chunk_id`s are infeasible by construction (each input piece gets its own call; the loop assigns `chunk_id` from the input).
- **AC-9a** Scratchpad capture: when the model includes a `<scratchpad>...</scratchpad>` block, the extracted text is stored verbatim in the entry's `scratchpad` field of `-relevance.json`. When omitted, `scratchpad is None` (JSON `null`). Verified by inspecting cold-run output for both test fixtures.
- **AC-10** Eval-call exception handling: forcing an exception in `evaluate_chunks` does not abort `path2chunks`; the document is processed with all pieces kept; one `"CHUNK"` log line is emitted.
- **AC-11** Async correctness: all callers of `path2chunks` await it correctly; running `python _nemo.py --chunk-only` produces no `RuntimeWarning: coroutine ... was never awaited` warnings.
- **AC-12** Schema stability: the JSON keys of `-chunks.json` and `-logic-chunks.json` are identical (set comparison) regardless of filter state. Only `source_chunk_ids` *content* differs.
- **AC-13** Logging: filter-on run emits exactly one summary line per document (`<doc>: <kept>/<total> pieces kept (filtered <d>; unsure <h>)`); exactly one cache-hit line per cache hit; the right number of validation warnings.

Spot-checks against truth:

- **SC-1** TBF000011: chunk 0 receives `score ∈ {0, 0.5}`; chunks 1-5 receive `score = 1`. Resulting `-logic-chunks.json` has logical chunks whose `source_chunk_ids` ⊆ [1, 2, 3, 4, 5].
- **SC-2** TBF000131: a manual scan of the resulting `-logic-chunks.json` shows zero references-list, FHWA-disclaimer, contact-info, availability-notice, or quality-assurance content in any `texts[].text`.
- **SC-3** False-negative budget on `score=1` (the failure mode under strict `> 0.5`): zero `truth=1 → model=0` errors; minimize `truth=1 → model=0.5` errors via prompt iteration. Track each mismatch.

## Decisions flagged

- **Filter at the recursive level, not the logical level.** Recursive pieces are smaller and more uniform; logical chunks already mix boilerplate with content (TBF000131 logical chunk 0). Filtering before grouping prevents mixed chunks and produces cleaner boundaries downstream.
- **Strict keep policy `> 0.5` (only `1` survives).** Biases toward precision. The `0.5` label is preserved as a distinct value (rather than going binary 0/1) so we get telemetry on model uncertainty and can revisit the threshold without re-running the eval.
- **Single LLM call per document, all pieces in one prompt.** Full-doc context lets the model distinguish "tail of document references" from "in-content citation" — distinguishing these cleanly is hard without surrounding context. Trade-off: scaling concern for very long documents; falls back to windowed eval if budget exceeded.
- **Per-piece eval, not bulk full-doc eval.** Empirically the bulk approach (all pieces tagged into one LLM call) mislabeled 4–6 pieces of body content as "References section" because the model used document context to over-extend the noise classification backward. Per-piece eval forces each chunk to be judged on its own content, which is sufficient because boilerplate categories (sponsor lists, references, disclaimers, mission statements, distribution notices, QA statements) are format-recognizable in isolation. Trade-offs accepted: ~5× cost (still ~$0.015/doc, $1.50/100 docs), N round-trips bounded by `relevance_concurrency` (default 8 → ~5–10 s wall time on an 84-piece doc). The wrapper Pydantic model (`RelevanceResponse`) and coverage soft-validation (`_reconcile_relevance`) are both eliminated; the per-piece schema (`RelevanceJudgment`) is simpler.
- **Chain-of-thought prompt with tag-based JSON output, not OpenAI Structured Outputs.** The new prompt at `prompts/nemo_eval-02.txt` instructs the model to reason in `<scratchpad>` tags before emitting `<json>`. This is incompatible with `client.beta.chat.completions.parse(response_format=...)` which requires pure JSON output. We use plain `client.chat.completions.create()` and extract the `<json>` block by regex, then validate via `RelevanceJudgment.model_validate_json()`. Trade-offs accepted: API-boundary enforcement is lost (model could return malformed JSON, missing tags, or off-enum scores), but the per-piece exception handler catches all such cases and falls back to `score=1.0`. Gain: the `<scratchpad>` reasoning is captured and stored in `-relevance.json`, which is useful for debugging and prompt iteration. Aligns with the prompt's intentional chain-of-thought design.
- **Direct OpenAI API for the eval call, not `BaseLLM.run_chain`.** The eval call is OpenAI-only, even when `[llm].provider` is Google or Ollama. Trade-off accepted to keep the eval call's prompt template and parsing logic separate from `BaseLLM`'s LangChain-wrapped path. Aligns with `extract_artifacts.py` v4's approach.
- **Pydantic confined to `evaluate_chunks` only — other `_nemo.py` LLM calls keep `BaseLLM.run_chain` + manual JSON.** `extract_artifacts`, `generate_qa_pairs`, and `evaluate_qa_pairs` are not migrated. Justified because `score` is a true closed enum where schema enforcement is load-bearing; the other three methods produce open-shaped content (variable-length artifact lists, free-text descriptions, prose answers) where Pydantic would force premature structure. Two patterns coexist in `_nemo.py` after this PR; that tension is accepted as the price of a narrow change. Migrating the other three is a separate decision with its own scope.
- **Soft-validation default = `1.0` for per-piece exceptions only.** Coverage gaps, duplicates, and out-of-range chunk_ids are eliminated by per-piece fan-out construction. The only remaining default is "this single piece's call raised" → `score=1.0` (preserves recall under the strict policy). Logged but not punitive. Out-of-set scores are impossible by construction.
- **Bounded concurrency (default 8), not unbounded `asyncio.gather`.** With 84 pieces and unbounded fan-out we'd hit OpenAI rate limits on large documents. `Semaphore(8)` is conservative enough for `gpt-4o-mini`'s tier and small enough to keep wall time reasonable (~5–10 s per 84-piece doc). Configurable via `[chunking].relevance_concurrency` for operators on different rate-limit tiers.
- **Mode 3 only in v1.** Modes 1 and 2 have different unit sizes (final pieces; 50-tok pre-pieces) and different cache shapes; designing the filter for them is a separate task. The flag is ignored (with a warning) for non-mode-3 to prevent silent inactivity.
- **Hardcoded `gpt-4o-mini` at `temperature=0.0`.** Matches `extract_artifacts.py` v4. Cost is negligible (~$0.003/doc on TBF000131-sized inputs) and independent of `[llm].model`. Configurability (`[chunking].relevance_model` / `.relevance_temperature`) deferred until a use case emerges.
- **No CLI flag.** TOML-only. `--relevance-filter` would clutter argparse for a feature that's expected to be set per-environment, not per-invocation.
- **No prompt-version marker on `-relevance.json`.** Operators force-regenerate via `self.overwrite = True` when the prompt changes. Same convention as the rest of the pipeline.
- **Refactor `path2chunks` to drive recursive split inline.** Previously it called `self.chunker.split(raw_text)` and read back `last_recursive_pieces` / `last_source_indices`. The filter pathway needs explicit recursive pieces *before* logical grouping, so `path2chunks` now calls `self.chunker.recursive.split(raw_text)` directly and uses `group_kept_pieces` for grouping. Behavior with `kept_indices` set to full coverage is equivalent to the previous implementation.
- **`HybridLogicalChunker.split` kept as a public method.** It remains the entry point used outside `path2chunks` (e.g. ad-hoc scripting); behavior unchanged. The filter pathway bypasses it in favor of `group_kept_pieces` for explicit mask handling.
