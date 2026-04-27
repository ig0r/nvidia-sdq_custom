# Plan: Logical (LLM-based) Chunking

Add a second, configurable chunking method (`LLMSemanticChunker`) alongside the existing recursive character chunking used in `_nemo.py::QAGenerator.path2chunks`. Method is selected via TOML config.

## Config (`cfg/nemo.toml`)

New `[chunking]` section. `[embedding].chunk_size` / `[embedding].recursive_overlap` stay for the embedder; `path2chunks` switches to reading from `[chunking]`.

```toml
[chunking]
method = "recursive"          # "recursive" | "logical"
chunk_size = 256              # recursive target; also assembly cap for logical
recursive_overlap = 50        # recursive overlap; also pre-split size for logical
logical_presplit_tokens = 50  # size of each tagged piece
logical_window = 40           # # of tagged pieces per LLM call
logical_stride = 30           # slide amount (window - carryover)
```

## Code changes

### New file `aisa/parse/chunkers.py`

- `class RecursiveTextChunker` — wraps `RecursiveCharacterTextSplitter` with the new `chunk_size` / `overlap` / `get_token_count`. Mirrors current `path2chunks` splitter behavior exactly.
- `class LLMSemanticChunker` — takes a `BaseLLM` + chunking cfg. `.split(text)`:
  1. Pre-split with `RecursiveCharacterTextSplitter(chunk_size=logical_presplit_tokens, overlap=0, length_function=get_token_count)`.
  2. Tag each piece as `<start_chunk_{i}>{text}<end_chunk_{i}>`.
  3. Slide a window of size `logical_window` with stride `logical_stride`. Each window invokes the `nemo_logical-chunk` prompt and expects a JSON response `{"split_after": [i, j, k]}`.
  4. Validate indices: strictly increasing, within window range. On parse/validation failure, log `CHUNK` warning and force a split at the window end.
  5. Assemble by concatenating pre-split pieces between split points. If an assembled chunk exceeds `chunk_size * 2` tokens, re-split it with the recursive splitter as a safety net.
- `def get_chunker(chunk_cfg, llm=None)` — factory returning one of the two based on `chunk_cfg["method"]`.

### Untouched: `aisa/parse/chunk.py`

The existing `RecursiveChunker` (token-budgeted LLM batcher used at `_nemo.py:194`) keeps its name. The name collision with the new `RecursiveTextChunker` is intentional — the two do different things and the existing call site must not change.

### `_nemo.py`

- `QAGenerator.__init__` accepts `chunk_cfg: dict`; builds `self.chunker = get_chunker(chunk_cfg, self.llm)`.
- `path2chunks` replaces the inline splitter block (current `_nemo.py:155-169`) with `raw_chunks = self.chunker.split(raw_text)`. Table/image stripping stays where it is.
- `self.chunk_dir` becomes `doc-chunks_{size}_{method}` so switching methods doesn't reuse the wrong cache.
- `main()` passes `cfg["chunking"]` into `QAGenerator`.

## New prompt file: `prompts/nemo_logical-chunk.txt`

Follows the style of the existing three prompts — single `{tagged_text}` variable, numbered instructions, explicit output contract. Uses JSON output so existing `json_mode=True` path works without special-casing.

```
You will receive a sequence of short text chunks wrapped in <start_chunk_N>...<end_chunk_N> tags.
Your job is to identify the positions where the text should be split into semantically coherent sections.

TAGGED TEXT:
{tagged_text}

INSTRUCTIONS:
1. A split should occur where the topic, subject, or logical flow meaningfully shifts.
2. Do NOT split inside a coherent paragraph, list, or tightly-coupled sequence of ideas.
3. Splits are expressed by the chunk index AFTER which the break occurs (the chunk with that index ends the preceding section).
4. Return at least one split unless the entire window is a single coherent unit.
5. Indices must be strictly increasing and within the range of the provided chunks.

Return a JSON object of the form: {"split_after": [3, 7, 12]}
```

## CLAUDE.md updates

- "Prompts" bullet: add `nemo_logical-chunk` as required when `[chunking].method = "logical"`.
- Architecture section: short note on the new `[chunking]` section and `aisa/parse/chunkers.py`; clarify the `RecursiveChunker` (batcher) vs `RecursiveTextChunker` (splitter) distinction.

## Tradeoff

Logical chunking adds roughly `ceil(doc_tokens / (presplit_tokens * stride))` extra LLM calls per doc before Stage 0.2. Better semantic boundaries (and likely better retrieval eval), but non-trivial cost uplift on long corpora. Recursive remains the default.

## Open decisions before implementing

- Confirm JSON output format for the logical-chunk prompt (vs plaintext `split_after: ...`).
- Confirm default config values (window 40 / stride 30 / presplit 50).
- Whether to drop the sliding window and require each doc fit in a single LLM call (simpler, but fails on long docs).

---

# Mode 3: `random_logical` — logical chunks built from recursive intermediates

Adds a third chunking mode where the LLM groups already-recursively-split chunks into semantic clusters. Both the recursive intermediate and the final logical chunks are written to disk for inspection.

## Config additions

```toml
[chunking]
method = "recursive"   # "recursive" | "logical" | "random_logical"
# (existing logical_* keys unchanged)
hybrid_window = 8      # 8 × 256 ≈ 2048 tok per LLM call (fits max_input_tokens=3000)
hybrid_stride = 6      # overlap = 2 pieces between windows
```

The mode reuses `chunk_size` and `recursive_overlap` for the recursive pre-stage. `logical_presplit_tokens` is unused.

## Code changes

In `aisa/parse/chunkers.py`:

1. **Extract shared helpers** so logic is not duplicated between `LLMSemanticChunker` and the new class:
   - `_llm_split_decisions(llm, prompt_template, pieces, window, stride) -> list[int]` — runs windowed LLM calls, validates each response, returns sorted/deduplicated split indices.
   - `_assemble_with_overlap_trim(pieces, split_points, has_overlap) -> tuple[list[str], list[list[int]]]` — groups pieces by split points; when `has_overlap=True`, strips shared suffix-prefix between consecutive pieces in each group (mirrors `_nemo.py::_trim_overlap_for_context` style). Returns final chunks and source-piece indices per chunk.
   - `_shared_suffix_prefix_len(a, b)` — small helper, duplicated locally to avoid `_nemo.py` import cycle.

2. **Refactor `LLMSemanticChunker`** to call those helpers with `has_overlap=False`. No behavior change.

3. **New `HybridLogicalChunker`** — composes a `RecursiveTextChunker` (chunk_size, recursive_overlap) for the pre-stage and runs the LLM windowing on the resulting pieces. `has_overlap=True` for assembly. Validates `hybrid_window × chunk_size <= [llm].max_input_tokens` at construction time. Exposes `last_recursive_pieces: list[str]` and `last_source_indices: list[list[int]]` for `path2chunks` to write the intermediate file. Reuses the same `nemo_logical-chunk` prompt as mode 2.

4. **`get_chunker`** — adds the `"random_logical"` branch.

## File outputs

Cache dir: `{output_dir}/doc-chunks_{chunk_size}_random_logical/`

Per document:
- `{doc_id}-chunks.json` — recursive intermediate, schema unchanged: `{doc_id, parsed_file, texts: [{text, chunk_id, tokens}], images, tables}`.
- `{doc_id}-logic-chunks.json` — final logical chunks; `texts` entries gain a `source_chunk_ids: list[int]` field listing which recursive `chunk_id`s each logical chunk was assembled from.

## `_nemo.py::path2chunks` changes

- Cache hit check: when `method == "random_logical"`, key on `-logic-chunks.json` (the final). If present, read `texts` and return.
- After running `self.chunker.split(raw_text)`:
  - For mode 3: write the recursive intermediate (`self.chunker.last_recursive_pieces`) to `-chunks.json`, and the final logical chunks (with `source_chunk_ids` derived from `self.chunker.last_source_indices`) to `-logic-chunks.json`.
  - For modes 1 and 2: unchanged.
- `self.doc_paths[file_path]` continues to point at `-chunks.json` for all modes — downstream stages derive their filenames via `.replace("-chunks.json", "-artifacts.json")`, which works regardless of method.

## Caching strategy (option A)

Per-mode cache. Mode 3 always recomputes its recursive intermediate; it does NOT read mode 1's `doc-chunks_256_recursive/` cache. Each method's directory is fully self-contained — no cross-mode coupling, no order dependency.

## Tradeoff

LLM picks boundaries at 256-token granularity (vs 50 in mode 2). Coarser decisions but more context per piece. Re-split safety net at `2 × chunk_size = 512` tokens fires when LLM groups 3+ recursive pieces (since `3 × ~206 effective ≈ 618` tokens). Source-index mapping is imprecise for re-split chunks (the entire pre-cap source list is attached to every sub-piece).
