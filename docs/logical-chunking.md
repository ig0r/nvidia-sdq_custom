# Logical Chunking

Two LLM-driven chunking modes are available via `[chunking].method` in `cfg/nemo.toml`:

- **`logical`** — pre-splits the document into small ~50-token pieces, asks the LLM where to draw boundaries. Implemented by `aisa/parse/chunkers.py::LLMSemanticChunker`.
- **`random_logical`** — pre-splits the document with the recursive splitter (`chunk_size=256, recursive_overlap=50`), then asks the LLM to group those recursive pieces into semantic clusters. Implemented by `aisa/parse/chunkers.py::HybridLogicalChunker`. Saves the recursive intermediate as `-chunks.json` and the final logical chunks (with `source_chunk_ids` provenance) as `-logic-chunks.json` in the same per-mode directory.

Both modes share the `nemo_logical-chunk` prompt and the same windowing / response-validation helpers (`_llm_split_decisions`, `_assemble_with_overlap_trim`).

## Parameters

All tunable in `cfg/nemo.toml` under `[chunking]`:

| Parameter | Default | Used by | Meaning |
|---|---|---|---|
| `chunk_size` | 256 | all | target tokens per final chunk; safety cap for logical/random_logical = `2 × chunk_size` |
| `recursive_overlap` | 50 | recursive, random_logical | overlap tokens between recursive pieces; also used by the safety re-split |
| `logical_presplit_tokens` | 50 | logical | size of each tagged pre-split piece (tokens) |
| `logical_window` | 40 | logical | pre-pieces packed into one LLM call |
| `logical_stride` | 30 | logical | how far each window advances |
| `hybrid_window` | 8 | random_logical | recursive pieces packed into one LLM call |
| `hybrid_stride` | 6 | random_logical | how far each window advances |

Derived: `overlap_between_windows = window − stride` pieces.

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

## Mode `random_logical` — recursive pre-split + LLM grouping

### The two-pass idea

Mode 1 (`recursive`) cuts blindly by token budget. Mode 2 (`logical`) asks the LLM at fine grain (~50-token pre-pieces) — accurate but many calls. Mode 3 is a hybrid: pre-split into **chunk-sized** pieces first, then have the LLM only decide *which adjacent pieces belong together*.

### At a glance

```
                  ┌─────────────────────────────────────────────┐
INPUT             │  full markdown text (tables/images stripped)│
                  └─────────────────────┬───────────────────────┘
                                        │
                                        ▼
PASS 1:                       RecursiveTextChunker
recursive pre-split           chunk_size=256, recursive_overlap=50
                                        │
                                        ▼
                    ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
recursive pieces →  │P0│P1│P2│P3│P4│P5│P6│P7│P8│P9│Pa│Pb│   ← saved as -chunks.json
                    └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
                     ~256 tok each, 50-tok tails overlap their neighbor
                                        │
                                        ▼
PASS 2:               sliding LLM windows  (hybrid_window=8, hybrid_stride=6)
LLM grouping
                    Window A (pieces 0..7)
                    ┌──┬──┬──┬──┬──┬──┬──┬──┐
                    │P0│P1│P2│P3│P4│P5│P6│P7│
                    └──┴──┴──┴──┴──┴──┴──┴──┘   nemo_logical-chunk prompt:
                                               "split_after": [2, 5]
                                 │
                                 │  stride 6 →
                                 ▼
                          Window B (pieces 6..b)
                          ┌──┬──┬──┬──┬──┬──┐
                          │P6│P7│P8│P9│Pa│Pb│   "split_after": [9]
                          └──┴──┴──┴──┴──┴──┘
                                        │
                                        ▼
                       union of split decisions: {2, 5, 9}
                                        │
                                        ▼
PASS 3:             groups: [P0..P2] [P3..P5] [P6..P9] [Pa..Pb]
assemble            join each group; trim shared suffix↔prefix overlap
                    (because recursive_overlap > 0)
                                        │
                                        ▼
OUTPUT              ┌────────────────┬───────────────┬──────────────────┬──────────────┐
final chunks  →     │ logical chunk0 │ logical chunk1│ logical chunk2   │ logical chunk3│
                    │ src=[0,1,2]    │ src=[3,4,5]   │ src=[6,7,8,9]    │ src=[a,b]     │
                    └────────────────┴───────────────┴──────────────────┴──────────────┘
                    saved as -logic-chunks.json (with source_chunk_ids provenance)
```

### Why it's "hybrid"

- **Recursive** does the heavy lifting (cheap, deterministic, respects `chunk_size`).
- **LLM** is only asked the cheap question — *"of these 8 contiguous chunks, where does the topic break?"* — never to write text.
- **Window/stride overlap** (`hybrid_window − hybrid_stride` pieces re-seen, defaults 8/6 → 2) gives each boundary two chances to be voted on; decisions union together, so a missed split in window A can still be caught in window B.

### Two outputs, by design

- `{doc}-chunks.json` — the recursive intermediate (so you can inspect what the LLM saw).
- `{doc}-logic-chunks.json` — the final logical chunks, each carrying `source_chunk_ids` pointing back at the recursive pieces they came from. Stage 1's `pos_doc` resolution and hard-neg mining all key off the *logical* chunks; the provenance is just for debuggability.

### Step-by-step (detailed)

The pre-split step is the standard recursive splitter, so each "piece" the LLM sees is already a 256-token recursive chunk with 50-token overlap. The LLM decides which contiguous groups of those pieces form a coherent logical section.

```
STEP 1 — pre-split with RecursiveCharacterTextSplitter(256, overlap=50)
─────────────────────────────────────────────────────────────────────

  ┌──────────┬──────────┬──────────┬──────────┬──────────┬─ …
  │   r0     │   r1     │   r2     │   r3     │   r4     │   …    each ≈ 256 tok
  │ ~256 tok │ ~256 tok │ ~256 tok │ ~256 tok │ ~256 tok │
  └──────────┴──────────┴──────────┴──────────┴──────────┴─ …
       └────50 tok overlap with neighbor on each side────┘

  Saved as -chunks.json (intermediate, for inspection)


STEP 2 — pack recursive pieces into LLM windows (defaults: window=8, stride=6)
─────────────────────────────────────────────────────────────────────────────

  piece index:   0          6     7          13    14
                 │          │     │          │     │
  Window 1 ▶ ╔════════════════════╗                       (pieces 0..7, 8 pieces ≈ 2048 tok)
             ║   8 recursive      ║                       1 LLM call
             ║   pieces           ║
             ╚════════════════════╝
                       ◀── stride = 6 ──▶
  Window 2 ▶            ╔════════════════════╗            (pieces 6..13, overlap 2)
                        ║   8 recursive      ║            1 LLM call
                        ║   pieces           ║
                        ╚════════════════════╝


STEP 3 — assemble logical chunks; strip the 50-token overlap between consecutive pieces
───────────────────────────────────────────────────────────────────────────────────────

  LLM returns split_after: [i, j, ...] per window → merge → sort → dedup

  e.g. boundaries at piece indices [1, 4]  produces three logical chunks:

  group 1: pieces [0, 1]   →  text = r0 + r1[after_overlap_with_r0:]    sources=[0,1]
  group 2: pieces [2, 3, 4] →  text = r2 + r3[after_ovl:] + r4[after_ovl:]  sources=[2,3,4]
  group 3: pieces [5, ...]  →  ...                                       sources=[5,...]

  Saved as -logic-chunks.json with source_chunk_ids = [0,1] / [2,3,4] / ...
```

The overlap-stripping uses the same shared-suffix-prefix detection as `_nemo.py::_trim_overlap_for_context`. Each logical chunk's `source_chunk_ids` field maps it back to the recursive `chunk_id`s that fed it — that's the "see how they're constructed" view.

### Guardrails worth knowing

- `hybrid_window × chunk_size ≤ [llm].max_input_tokens` is enforced at construction in `chunkers.py:213-218` — otherwise a window won't fit in one LLM call.
- If the LLM returns no valid splits for a window, `_llm_split_decisions` falls back to splitting at the window's last piece (`chunkers.py:73-78`) — the worst case degrades to "behave like recursive".
- Unlike mode 2, **no output-side `_enforce_size_cap`** — see below.

### No output-side cap (the LLM's grouping is final)

Mode 3 does not apply any output-side size cap. Whatever the LLM groups is what you get; `source_chunk_ids` is always exact. The upper bound on a logical chunk's size is set entirely by the input window: at most `hybrid_window × chunk_size` tokens (minus accumulated overlap-trim between consecutive recursive pieces). With defaults (`hybrid_window=8, chunk_size=256, recursive_overlap=50`) that's roughly **1,700 tokens** worst case.

Why no cap: capping the output silently would override the LLM's grouping decisions — exactly what mode 3 was designed to enable. Downstream stages handle larger chunks correctly: `extract_artifacts` bundles by token budget, `mine_hard_negatives` embeds passages directly. If you want smaller chunks, lower `hybrid_window`.

If you ever switch to a short-context embedder (e.g., 512-token sentence-transformer models) at Stage 1, large logical chunks may be silently truncated by the embedder. That's a per-embedder concern handled outside chunking.

### Mode 2 size cap (unchanged)

Mode 2 (`logical`) keeps its `2 × chunk_size` safety cap and character-wise re-split. Mode 2's pre-pieces are 50 tokens, so without the cap a degenerate LLM response (no splits in a window) could produce one chunk of ~2000 tokens — too coarse for the design intent. Mode 2 has no source-mapping to lose, so character-wise re-splitting there is harmless.

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
