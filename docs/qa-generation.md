# QA Generation

Two question-generation flows live in `_nemo.py`. They share `path2chunks` (Stage 0.1) but differ in how chunks are bundled before the LLM is asked to write QA pairs.

| Flag | Bundling | Status |
|---|---|---|
| `--sdg` | Multiple chunks per bundle, packed by `RecursiveChunker` under `[llm].max_input_tokens` and de-overlapped by `_trim_overlap_for_context` | Full pipeline (artifacts → QA pairs → eval) |
| `--sdg-logical` | One **logical chunk** per bundle (1:1) | **Step 1 only** — bundles written, downstream stages deferred |

Both flows write into `{output_dir}/doc-chunks_{chunk_size}_{method}/` and use disjoint filename suffixes so they coexist in one output dir.

---

## `--sdg-logical` (logical-chunk flow, Step 1)

Generates `{doc_id}-logic-ctx.json` per document — one bundle per logical chunk — and stops there. Future steps will add `-logic-artifacts.json`, `-logic-qa_pairs.json`, `-logic-qa_eval.json`.

### Prerequisite
Set `[chunking].method` in `cfg/nemo.toml` to `"logical"` or `"random_logical"`. With `"recursive"` the command raises `ValueError` immediately, before any docs are touched.

### Invocations

```bash
# Step 1 only — bundle every logical chunk into -logic-ctx.json, then stop
.venv/bin/python _nemo.py --sdg-logical --cfg cfg/nemo.toml

# Override input/output dirs (same as the other tasks)
.venv/bin/python _nemo.py --sdg-logical \
  --cfg cfg/nemo.toml \
  --input_dir ./rawdata/parsed-techbriefs \
  --output_dir ./data/nemo_briefs_20260422

# Combine with chunking — useful for a fresh corpus where chunks aren't cached.
# --chunk-only runs first, --sdg-logical reads its output.
.venv/bin/python _nemo.py --chunk-only --sdg-logical --cfg cfg/nemo.toml

# Run both flows in one invocation — each writes its own files
.venv/bin/python _nemo.py --sdg --sdg-logical --cfg cfg/nemo.toml
```

### Output files

```
{output_dir}/doc-chunks_{chunk_size}_{method}/
  {doc_id}-chunks.json         ← always (recursive in random_logical, logical otherwise)
  {doc_id}-logic-chunks.json   ← only in random_logical mode
  {doc_id}-logic-ctx.json      ← written by --sdg-logical
```

`-logic-ctx.json` wraps the entries with `doc_id` so the file self-identifies its source document:
```json
{
  "doc_id": "ABC_123",
  "contexts": [
    {"chunks": [{"chunk_id": 0, "text": "...", "tokens": 123}], "tokens": 123},
    {"chunks": [{"chunk_id": 1, "text": "...", "tokens": 456}], "tokens": 456}
  ]
}
```
Single-element `chunks` list per entry, `tokens` mirrors the chunk's own count. In `random_logical` mode each chunk additionally carries `source_chunk_ids: list[int]`. Note: this diverges from the existing `-ctx.json` (a bare JSON array); schema parity for downstream reuse will be revisited in Step 2.

### Re-running / idempotency

`_build_logical_contexts` skip-writes when `Path(out_path).exists() and not self.overwrite`. To force regeneration:
- Delete the `-logic-ctx.json` files, **or**
- Set `self.overwrite = True` in `QAGenerator.__init__` (`_nemo.py:117`).

There is no CLI flag for overwrite.

### Log signals (`CHUNK` level)

```
... | CHUNK | _build_logical_contexts - <file>.md: cache hit -> .../-logic-ctx.json
... | CHUNK | run_sgd_logical_pipeline - <file>.md: N logical-context entries -> .../-logic-ctx.json
```
Plus one warning per oversized bundle:
```
... | CHUNK | _build_logical_contexts - <file>.md: chunk_id=K tokens=T > max_input_tokens=B
```
The warning is informational — Step 1 passes oversized bundles through. Hard handling is deferred to Step 2.

---

## `--sdg` (bundled flow, full pipeline)

Runs four LLM stages per document: bundle → artifact extraction → QA generation → LLM-as-judge eval, plus the corpus-wide aggregate. Driven by `_nemo.py::run_sgd_pipeline`; prompts live under `prompts/nemo_*.txt`.

```bash
.venv/bin/python _nemo.py --sdg --cfg cfg/nemo.toml
```

### How a question is built — stage by stage

Once `path2chunks` has produced `{doc_id}-chunks.json`, the SDG pipeline runs three more LLM stages per document. Every stage caches its output and skip-writes when present.

#### Stage 1 — Bundle chunks into LLM-sized contexts (`extract_artifacts`)
The chunk list is handed to `aisa/parse/chunk.py::RecursiveChunker(custom_input=chunks, max_input_tokens=...)` which packs **consecutive chunks** until they would overflow the LLM's input budget. Each pack becomes one **bundle**.

Then `_trim_overlap_for_context` walks each bundle and strips the suffix/prefix overlap between neighboring chunks (the recursive splitter introduced overlap on purpose for retrieval; inside one prompt it's wasted tokens).

The de-overlapped bundles are written to `-ctx.json`. So we now have N bundles, each a list `[{chunk_id, text, tokens}, …]` totaling ≤ `[llm].max_input_tokens`.

#### Stage 2 — Extract "facts" per bundle (`extract_artifacts`, same call)
Each bundle's joined text is sent through `nemo_artifacts`. The LLM returns up to `max_artifacts` items in 8 buckets: `key_concepts`, `relationships`, `themes`, `entities`, `processes`, `insights`, `technical_terms`, `contextual_factors`. Output → `-artifacts.json`.

`get_fact_blocks` formats each bundle's artifacts into an XML-ish `<key_concepts>… </key_concepts><relationships>… </relationships>…` string — one **`facts_block`** per bundle.

`get_ctx_blocks` reads `-ctx.json` and re-formats each bundle as `=== Section 1 ===\nSegment {chunk_id}: {text}\n…` — one **`context_block`** per bundle.

So for each bundle we now have a paired `(facts_block, context_block)`.

#### Stage 3 — Generate QA pairs (`generate_qa_pairs`)
For every bundle, one call to `nemo_qa-gen` is made with:

- `facts_block` — *what to ask about* (the artifacts).
- `context_block` — *what evidence the LLM can draw from* (the actual segments, with their `chunk_id`s).
- Hardcoded counts from `QAGenerator.__init__` (`_nemo.py:118-132`):
  - **Query types** (each question gets one): `multi_hop` (5), `structural` (5), `contextual` (5).
  - **Reasoning types** (orthogonal axis): `factual` / `relational` / `inferential` / `temporal` / `procedural` (3 each), `visual` / `causal` (0).
  - `min_hops=2`, `max_hops=3`, `min_complexity=3`, `num_pairs=15`.

The prompt instructs the model to:
- Ask questions that require **connecting multiple segments** (no shallow lookups).
- Never reference "the transcript" / "the context" — questions must read standalone.
- Tag each question with both a `query_type` AND a `reasoning_type` (orthogonal fields).
- Record `segment_ids` — the chunk IDs that are the **source material** for the question. These become the positives (`pos_doc`) downstream in `--prep`.
- For multi-hop, include `hop_contexts` listing per-hop segment IDs and a summary.

Output → `-qa_pairs.json`: one entry per bundle, each with a `pairs` array of ~15 `{question, answer, query_type, reasoning_type, question_complexity, segment_ids, hop_count, hop_contexts}` records.

To change the counts, edit the constants in `QAGenerator.__init__` — they are not exposed in TOML.

#### Stage 4 — Judge them (`evaluate_qa_pairs`)
Each bundle's QA pairs + the same `context_block` go to `nemo_eval`. The judge scores each pair. Pairs with `overall.score < 7.0` are dropped later in Stage 1 of `--prep` (`filter_and_convert`, threshold configurable via the `quality_threshold` argument).

### Data flow

```
chunks (-chunks.json)
  └─► RecursiveChunker bundling + overlap trim
        ├─► -ctx.json   ── format ──► context_block (segments w/ chunk_ids)
        └─► nemo_artifacts ── -artifacts.json ── format ──► facts_block
              └─► nemo_qa-gen(facts_block, context_block, counts)
                    └─► -qa_pairs.json (questions tagged with segment_ids)
                          └─► nemo_eval ── -qa_eval.json (filter < 7.0 in --prep)
```

The crucial design point: `facts_block` tells the LLM *what's interesting* (distilled), while `context_block` keeps the raw segments with their `chunk_id`s in view so the model can both ground answers and emit accurate `segment_ids` — those IDs are what stitches each question back to its positive chunks downstream.

### Outputs (per doc, in the same chunk dir)
```
{doc_id}-ctx.json        bundles after RecursiveChunker packing + overlap trim
{doc_id}-artifacts.json  per-bundle artifact extraction (nemo_artifacts)
{doc_id}-qa_pairs.json   per-bundle QA pairs (nemo_qa-gen)
{doc_id}-qa_eval.json    per-bundle judge scores (nemo_eval)
{doc_id}-sdg.json        per-doc combined record
```
Plus `{output_dir}/full_sdg_output.json` — the corpus-wide aggregate consumed by `--prep`.
