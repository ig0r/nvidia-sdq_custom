# Logical Chunking

LLM-driven semantic chunking used when `[chunking].method = "logical"` in `cfg/nemo.toml`. Implemented in `aisa/parse/chunkers.py::LLMSemanticChunker`.

## Parameters

All tunable in `cfg/nemo.toml` under `[chunking]`:

| Parameter | Default | Meaning |
|---|---|---|
| `logical_presplit_tokens` | 50 | size of each pre-split piece (tokens) |
| `logical_window` | 40 | pieces packed into one LLM call |
| `logical_stride` | 30 | how far each window advances |
| `chunk_size` | 256 | soft ceiling — any final chunk > `2 × chunk_size` is safety-re-split |
| `recursive_overlap` | 50 | overlap used by the safety re-split |

Derived: `overlap_between_windows = logical_window − logical_stride = 10` pieces.

## How it works

```
Config (cfg/nemo.toml [chunking])
  logical_presplit_tokens = 50   ──▶ size of each piece (tokens)
  logical_window          = 40   ──▶ pieces per LLM call
  logical_stride          = 30   ──▶ how far each window advances
  overlap = window − stride = 10 ──▶ pieces shared by neighboring windows


STEP 1 — pre-split the document into fixed-size pieces (~50 tok each)
─────────────────────────────────────────────────────────────────────

   raw text
      │
      ▼  RecursiveCharacterTextSplitter(chunk_size=50, overlap=0)
      │
      ▼
  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬─ …
  │ p0 │ p1 │ p2 │ p3 │ p4 │ p5 │ p6 │ p7 │ p8 │ p9 │p10 │p11 │p12 │   …
  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴─ …
   each piece is tagged:  <start_chunk_0>…<end_chunk_0><start_chunk_1>…<end_chunk_1>…


STEP 2 — pack into sliding windows; ONE LLM call per window
────────────────────────────────────────────────────────────

   piece index:   0          30         40         60         69         99
                  │          │          │          │          │          │
  Window 1 ▶ ╔══════════════════════════╗                                    (pieces  0..39)
             ║  40 pieces ≈ 2000 tokens ║                                    1 LLM call
             ╚══════════════════════════╝
                   ◀──── stride = 30 ────▶
  Window 2 ▶            ╔══════════════════════════╗                         (pieces 30..69)
                        ║  40 pieces ≈ 2000 tokens ║                         1 LLM call
                        ╚══════════════════════════╝
                                     ◀──── stride = 30 ────▶
  Window 3 ▶                                    ╔══════════════════════════╗ (pieces 60..99)
                                                ║  40 pieces ≈ 2000 tokens ║ 1 LLM call
                                                ╚══════════════════════════╝

                        ◀─ overlap = 10 ─▶     ◀─ overlap = 10 ─▶
                       (pieces seen by both      (pieces seen by both
                         W1 and W2)                W2 and W3)


STEP 3 — merge splits from all windows, assemble final chunks
──────────────────────────────────────────────────────────────

   per window:  LLM returns  {"split_after": [i, j, k, …]}
                    │
                    ▼  validate: int, strictly increasing, within [win_start, win_end)
                    ▼  on failure: warn + force split at win_end − 1
                    │
   all windows  ──▶ union of valid indices ──▶ sort ──▶ deduplicate
                    │
                    ▼
   final chunks = concat(pieces between consecutive split points)
                    │
                    ▼  safety: any chunk > 2 × chunk_size (= 512 tok)
                    ▼          is re-split via RecursiveCharacterTextSplitter
                    │
   list[str] ──▶ path2chunks wraps each with {text, chunk_id, tokens}
```

## LLM call-count formula

For a document of `N` tokens, with defaults (`presplit=50`, `stride=30`):

```
pieces     = ceil(N / 50)
LLM calls  = ceil(pieces / 30)   (minus 1 if the last window would be < 2 pieces)
           ≈ ceil(N / 1500)
```

Examples:
- 3 KB doc (~750 tok) → ~15 pieces → **1 call** (one window covers the whole doc)
- 15 KB doc (~4 K tok) → ~80 pieces → **3 calls**
- 50 KB doc (~12 K tok) → ~240 pieces → **9 calls**

## Tuning knobs

- **Smaller `logical_presplit_tokens`** (e.g., 25) → finer-grained split decisions, more tag overhead, more calls.
- **Larger `logical_window`** → more context per decision but bumps against `[llm].max_input_tokens` (currently 3000). At `window=40, presplit=50` you're already near ~2080 tokens including tag overhead; don't push much higher without raising `max_input_tokens`.
- **Larger `logical_stride`** (closer to `logical_window`) → fewer calls, less boundary overlap, cheaper but riskier at window edges.
- **Smaller `chunk_size`** → safety re-split kicks in more aggressively on long LLM-chosen sections.

## Running chunking only

```bash
python _nemo.py --chunk-only --cfg cfg/nemo.toml \
  --input_dir ./path/to/md_corpus \
  --output_dir ./data/chunk_test
```

Flip `[chunking].method` between `"recursive"` and `"logical"` in the TOML to compare. Output lands at `{output_dir}/doc-chunks_{chunk_size}_{method}/{doc_id}-chunks.json`.
