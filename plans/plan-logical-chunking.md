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
