# `eval2.py` — base vs fine-tuned embedding eval on techbriefs

Single-script evaluator that asks every techbriefs question against an embedding-based
retrieval index of techbriefs answers, under two model configurations:

| Method   | Embedding model (default)                              | Index            |
|----------|--------------------------------------------------------|------------------|
| `rag`    | `BAAI/bge-base-en-v1.5` (base BGE)                     | Qdrant collection `techbriefs_answers` |
| `rag-ft` | `./examples/eval2/bge-base-pavements-matryoshka` (FT)  | Qdrant collection `techbriefs_answers_ft` |

Both methods write into the same on-disk Qdrant store (different collections),
so they coexist with no rebuild interference.

The fine-tune was trained on **pub242** Q&A pairs, so this is a held-out-domain
test of whether that fine-tune transfers to the **techbriefs** corpus
(40,137 QA pairs across 125 docs / 1,187 logical chunks).

Sibling script: `examples/eval2/predict_pub242_v2.py` runs the same model
comparison on pub242 and supports six methods (Qdrant + FAISS × baseline +
fine-tuned + Unsloth). `eval2.py` is its trimmed cousin focused on the two
methods that actually matter for the techbriefs comparison.

---

## 1. What it does

For each techbriefs question (or the first `N` if `-n N` is passed):

1. Embeds the question with the section's model (`encode_query` if the model
   exposes it, else `encode`).
2. Issues a Qdrant `query_points` call against the matching collection,
   retrieving `top_k + 1` candidates so the self-exclusion metrics still have
   `top_k` items after the self-match is removed.
3. Computes three metric families per `k ∈ match_at_k` (default `[1, 3, 5]`):
   - **`qid_top{k}`** — query's own `question_id` is in top-K (trivial
     self-retrieval check; no exclusion).
   - **`logic_chunk_top{k}`** — query's `u_logic_chunk_id` matches a hit's
     `u_logic_chunk_id` in the top-K **after self-exclusion** (drop the
     prediction whose `question_id` equals the query's).
   - **`ctx_top{k}`** — same shape, but on `u_ctx_id`.
4. Saves predictions + match flags to one JSON + one CSV per method.

Each saved prediction carries full provenance:

- `question_id` (the QA pair the answer originated from)
- `doc_id` (techbriefs document, e.g. `TBF000001_UKN000`)
- `u_ctx_id` (context window within the doc, e.g. `TBF000001_UKN000-ctx-0`)
- `u_logic_chunk_id` (logical chunk within the doc, e.g. `TBF000001_UKN000-logic-chunk-0`)
- `score` (cosine similarity)
- `answer_snippet` (first 200 chars of the retrieved answer)

This lets downstream analysis answer not just *"did it retrieve the right
answer?"* but *"which doc / logical chunk / context did it pull from?"*.

---

## 2. Quick start

```bash
cd /Users/igor/dev/llm/pavement-gpt/nvidia-pipeline/nvidia-sdq_custom

# 1) Build the base-model Qdrant collection (one-time; ~30–60 min on CPU, seconds on GPU).
.venv/bin/python eval2.py --setup-rag

# 2) Run the base-model eval.
.venv/bin/python eval2.py --batch-rag

# 3) Build the fine-tuned-model collection (separate, coexists in the same db_path).
.venv/bin/python eval2.py --setup-rag-ft

# 4) Run the fine-tuned-model eval.
.venv/bin/python eval2.py --batch-rag-ft
```

Smoke test on a tiny slice (build is full-corpus regardless of `-n`, so
`-n` only narrows the eval):

```bash
.venv/bin/python eval2.py --batch-rag -n 50
```

First base-model run downloads `BAAI/bge-base-en-v1.5` (~430 MB) into
`~/.cache/huggingface/`. The fine-tune lives in-repo at
`./examples/eval2/bge-base-pavements-matryoshka/`. **Subsequent runs of
`--batch-*` skip the build entirely** — the Qdrant collections persist on disk.

---

## 3. CLI

```
python eval2.py [--config PATH] [--top-k K] [-n N]
                (--setup-rag | --batch-rag | --setup-rag-ft | --batch-rag-ft)
```

| Flag             | Default        | Effect                                                              |
|------------------|----------------|---------------------------------------------------------------------|
| `--config`       | `./eval2.toml` | TOML config path (defaults to a sibling of the script).             |
| `--top-k`        | `[rag*] top_k` | Override retrieval depth. `match_at_k` clamps against it.           |
| `-n`             | (no limit)     | Limit the **eval** to the first N questions. Build is always full-corpus. |
| `--setup-rag`    | —              | Build/rebuild the base-model Qdrant collection.                     |
| `--batch-rag`    | —              | Run the base-model eval (auto-builds if collection missing).        |
| `--setup-rag-ft` | —              | Build/rebuild the fine-tuned-model Qdrant collection.               |
| `--batch-rag-ft` | —              | Run the fine-tuned-model eval (auto-builds if collection missing).  |

Exactly one of the four action flags must be passed; otherwise the CLI errors.

`-n` truncates the question list (first N) — there is no random-sample mode.
For unbiased smoke comparisons, the first N rows of the input JSON are
heavily concentrated in early documents (`TBF000001_…`), so prefer a
larger `-n` (e.g. 1000) before reading too much into headline numbers.

---

## 4. Configuration (`eval2.toml`)

```toml
[general]
input      = "./data/nemo_briefs_20260429/qa-gen-cluster/generated-questions_wo_context-c-eval.json"
db_path    = "./data/nemo_briefs_20260429/eval2/qdrant_db"
output_dir = "./data/nemo_briefs_20260429/eval2"
name       = "techbriefs"

[rag]
collection_name   = "techbriefs_answers"
embedding_backend = "sentence-transformers"
embedding_model   = "BAAI/bge-base-en-v1.5"
embedding_dim     = 768
encode_batch_size = 64
top_k             = 5
match_at_k        = [1, 3, 5]

[rag-ft]
collection_name   = "techbriefs_answers_ft"
embedding_model   = "./examples/eval2/bge-base-pavements-matryoshka"
embedding_dim     = 768
encode_batch_size = 64
top_k             = 5
match_at_k        = [1, 3, 5]
```

| Key                              | Purpose                                                                 |
|----------------------------------|-------------------------------------------------------------------------|
| `[general] input`                | JSON list of techbriefs QA pairs (queries + answer corpus, same file).  |
| `[general] db_path`              | Local on-disk Qdrant store. Both methods' collections live here.        |
| `[general] output_dir`           | Where `<name>_<method>_results.{json,csv}` are written.                 |
| `[general] name`                 | Filename prefix for outputs (defaults to `techbriefs`).                 |
| `[rag\|rag-ft] collection_name`  | Distinct Qdrant collection per method — switching methods doesn't rebuild. |
| `[rag] embedding_backend`        | `"sentence-transformers"` or `"ollama"`. Only `[rag]` honours this.     |
| `[rag\|rag-ft] embedding_model`  | HF model name OR a local path (resolved against CWD if relative).       |
| `[rag\|rag-ft] embedding_dim`    | Vector dimension. Must match the model.                                 |
| `[rag\|rag-ft] encode_batch_size`| Doc-encoding batch size during build. Tune for VRAM.                    |
| `[rag\|rag-ft] top_k`            | Predictions retained per query.                                         |
| `[rag\|rag-ft] match_at_k`       | K values for the match-at-K columns (always ≤ `top_k`).                 |

All path keys resolve **relative to the current working directory** (the
project root) if not absolute. This diverges from `eval.py`, which resolves
relative to the config file's parent. Always run `eval2.py` from the project
root (or use absolute paths in the TOML).

---

## 5. Outputs

### File layout (under `[general] output_dir`)

```
output_dir/
├── qdrant_db/                                      # local Qdrant store
│   ├── collection/techbriefs_answers/...           # base-model vectors + payloads
│   └── collection/techbriefs_answers_ft/...        # fine-tune vectors + payloads
├── techbriefs_rag_results.json
├── techbriefs_rag_results.csv
├── techbriefs_rag_ft_results.json
└── techbriefs_rag_ft_results.csv
```

Output filenames are derived as `{name}_{method.replace('-','_')}_results.{json,csv}`.

### JSON shape (per method)

```json
{
  "model": {
    "name":              "techbriefs",
    "method":            "rag",
    "version":           "v2",
    "eval_mode":         "self_retrieval",
    "doc_source":        "answers",
    "embedding_model":   "BAAI/bge-base-en-v1.5",
    "embedding_dim":     768,
    "embedding_backend": "sentence-transformers",
    "collection_name":   "techbriefs_answers",
    "num_points":        40137,
    "num_docs":          40137
  },
  "results": [
    {
      "question_id": "TBF000001_UKN000-ctx-0-q-0",
      "question":    "What is the main focus of this study?",
      "ground_truth": {
        "question_id":      "TBF000001_UKN000-ctx-0-q-0",
        "doc_id":           "TBF000001_UKN000",
        "u_ctx_id":         "TBF000001_UKN000-ctx-0",
        "u_logic_chunk_id": "TBF000001_UKN000-logic-chunk-0"
      },
      "predictions": [
        {
          "rank":             1,
          "question_id":      "TBF000001_UKN000-ctx-0-q-0",
          "doc_id":           "TBF000001_UKN000",
          "u_ctx_id":         "TBF000001_UKN000-ctx-0",
          "u_logic_chunk_id": "TBF000001_UKN000-logic-chunk-0",
          "score":            0.93,
          "answer_snippet":   "first 200 chars ..."
        }
      ],
      "matches": {
        "qid":         {"top1": true,  "top3": true,  "top5": true},
        "logic_chunk": {"top1": true,  "top3": true,  "top5": true},
        "ctx":         {"top1": true,  "top3": true,  "top5": true}
      }
    }
  ]
}
```

### CSV shape (one row per query)

```
question_id, question, gt_doc_id, gt_u_ctx_id, gt_u_logic_chunk_id,
pred_1_qid, pred_1_doc_id, pred_1_u_ctx_id, pred_1_u_logic_chunk_id, score_1,
... ×top_k ...,
qid_top1_match,         qid_top3_match,         qid_top5_match,
logic_chunk_top1_match, logic_chunk_top3_match, logic_chunk_top5_match,
ctx_top1_match,         ctx_top3_match,         ctx_top5_match
```

Missing predictions (rare — only when Qdrant returns fewer than `top_k`)
are padded with empty fields.

### Console summary (per method)

```
Done. 40137 questions processed ([rag] search).
  Top-1  qid: 32154/40137 = 0.8011   logic_chunk: 21902/40137 = 0.5457   ctx: 21902/40137 = 0.5457
  Top-3  qid: 36874/40137 = 0.9187   logic_chunk: 28471/40137 = 0.7094   ctx: 28471/40137 = 0.7094
  Top-5  qid: 38109/40137 = 0.9494   logic_chunk: 31002/40137 = 0.7724   ctx: 31002/40137 = 0.7724
Output JSON: data/nemo_briefs_20260429/eval2/techbriefs_rag_results.json
Output CSV:  data/nemo_briefs_20260429/eval2/techbriefs_rag_results.csv
```

(Numbers are illustrative — your actual values depend on the model.)

---

## 6. How the two methods differ internally

`rag` and `rag-ft` are structurally identical pipelines — same code path, same
Qdrant API calls, same metrics. Three things change between them:

| Aspect            | `rag`                              | `rag-ft`                                            |
|-------------------|------------------------------------|-----------------------------------------------------|
| Model             | HF name (downloaded on first use)  | Local fine-tuned ST checkpoint                      |
| Collection name   | `techbriefs_answers`               | `techbriefs_answers_ft`                             |
| Output basename   | `techbriefs_rag_results.{json,csv}`| `techbriefs_rag_ft_results.{json,csv}`              |

The two collections live side-by-side in the same `db_path`, so building
`rag-ft` does **not** invalidate `rag`'s vectors. You can rerun `--batch-rag`
after `--setup-rag-ft` and get identical results to before.

The shared eval loop (`_run_eval`) is invoked with a `retrieve_fn` closure
that captures the section's model + collection — so adding more sections
later is a small change (new TOML section + a new CLI flag pair).

---

## 7. Idempotency

`ensure_rag_db(cfg, section)` checks `client.collection_exists(...)` and
`points_count > 0`. If both are true and `--setup-*` is not passed, it logs:

```
[rag] DB ready: collection='techbriefs_answers', points=40137, dim=768
```

…and returns immediately. No re-encoding, no upsert. `--batch-*` calls
`ensure_*` with `force=False`, so the first batch run after a build is fast,
and `--setup-*` (which calls with `force=True`) is the only path that
re-encodes.

The Qdrant store at `[general] db_path` is the canonical state — delete the
`qdrant_db/collection/<name>/` subdirectory (or pass `--setup-*`) to force a
fresh build of one collection. The output JSON / CSV files are overwritten
on each `--batch-*` run.

---

## 8. Sanity checks

After a smoke run, verify:

```bash
# Result count matches what you asked for
jq '.results | length' data/nemo_briefs_20260429/eval2/techbriefs_rag_results.json

# Predictions carry the full provenance fields
jq '.results[0].predictions[0] | keys' data/nemo_briefs_20260429/eval2/techbriefs_rag_results.json
# → ["answer_snippet","doc_id","question_id","rank","score","u_ctx_id","u_logic_chunk_id"]

# Match families are present at the requested K values
jq '.results[0].matches' data/nemo_briefs_20260429/eval2/techbriefs_rag_results.json
# → {"qid":{"top1":..,"top3":..,"top5":..}, "logic_chunk":{...}, "ctx":{...}}

# Row alignment in CSV (header width == row width)
awk -F, '{print NF}' data/nemo_briefs_20260429/eval2/techbriefs_rag_results.csv | sort -u
# → exactly one value (39 with default top_k=5, match_at_k=[1,3,5])

# Inspect the Qdrant collection directly
.venv/bin/python -c "
from qdrant_client import QdrantClient
c = QdrantClient(path='data/nemo_briefs_20260429/eval2/qdrant_db')
print(c.get_collections())
print('rag points:', c.count('techbriefs_answers'))
print('rag-ft points:', c.count('techbriefs_answers_ft'))
"
```

`qid_top1` should be very high (~0.8+) for both models — the question's exact
answer is in the corpus, so self-retrieval is mostly trivial. `logic_chunk` and
`ctx` are the meaningful comparison points: they exclude the self-match before
scoring, so they reflect "does the embedder cluster siblings of the same chunk
near each other?".

---

## 9. Common operations

### Smoke test against a tiny slice

```bash
# Build runs the full corpus, but eval is bounded by -n.
# For a faster smoke, build once then eval on -n 100.
.venv/bin/python eval2.py --setup-rag      # one-time
.venv/bin/python eval2.py --batch-rag -n 100
```

### Compare base vs fine-tuned on the same K

```bash
.venv/bin/python eval2.py --batch-rag    --top-k 10
.venv/bin/python eval2.py --batch-rag-ft --top-k 10
# Diff the JSONs — same question_ids in the same order.
```

### Rebuild only one collection

```bash
# rebuild the FT collection without touching the base one
.venv/bin/python eval2.py --setup-rag-ft
```

### Switch to a different fine-tune

Edit `[rag-ft] embedding_model` (and `embedding_dim` if it differs), then:

```bash
.venv/bin/python eval2.py --setup-rag-ft   # vectors from the old FT are now incompatible
.venv/bin/python eval2.py --batch-rag-ft
```

### Switch to the Qwen3 fine-tune (1024-dim)

```toml
[rag-ft]
embedding_model = "./examples/eval2/finetuned-qwen3-embedding-pplx_20260409_011123/merged_checkpoints"
embedding_dim   = 1024
```

Then `--setup-rag-ft` to re-embed under the new dimension.

### Ollama backend for the base model

`[rag]` accepts `embedding_backend = "ollama"` (and `ollama_base_url`,
defaulting to `http://localhost:11434`). `[rag-ft]` is sentence-transformers
only — Ollama doesn't host arbitrary local fine-tunes.

### Inspect what got retrieved for a particular question

```bash
jq '.results[] | select(.question_id == "TBF000001_UKN000-ctx-0-q-0") | .predictions' \
   data/nemo_briefs_20260429/eval2/techbriefs_rag_results.json
```

---

## 10. Metrics — what they tell you

| Metric                  | Reads as                                                                    |
|-------------------------|-----------------------------------------------------------------------------|
| `qid_top{k}`            | "Did the embedder find the original answer to this question?" (trivial sanity check) |
| `logic_chunk_top{k}`    | "If we *hide* the original answer, does the embedder still surface a sibling answer from the same logical chunk?" |
| `ctx_top{k}`            | "If we hide the original, does the embedder surface a sibling from the same context window?" |

`qid` is high by construction (the question and its answer are paired in the
corpus). It's there as a sanity probe — sub-`0.5` `qid_top1` would indicate
something is wrong with the embedding pipeline.

`logic_chunk` and `ctx` (both with self-exclusion) are the two metrics worth
comparing across `rag` and `rag-ft`. They answer "did this fine-tune learn
representations that cluster topical siblings", which is what we actually want
the model to do for downstream RAG.

In the techbriefs corpus, `u_logic_chunk_id` and `u_ctx_id` happen to be 1:1
(1,187 unique values each) — so the two columns are usually equal. If you
later switch to a chunking config where they diverge (e.g. multiple ctx per
chunk), the difference becomes meaningful.

---

## 11. Troubleshooting

| Symptom                                                | Fix                                                                  |
|--------------------------------------------------------|----------------------------------------------------------------------|
| `Error: config file not found: ./eval2.toml`           | Pass `--config <path>` or run from the project root.                 |
| `Error: questions JSON not found: ...`                 | Check `[general] input` and your CWD. Paths are CWD-relative.        |
| `provide one of --setup-rag / --batch-rag / ...`       | The CLI requires exactly one action flag.                            |
| `ModuleNotFoundError: qdrant_client`                   | Run via `.venv/bin/python` — the system Python is missing deps.      |
| FT model errors (`OSError: ... not a folder`)          | The local FT path didn't resolve. Confirm `./examples/eval2/bge-base-pavements-matryoshka/` exists from CWD. |
| `RuntimeError: shape mismatch` after a model swap      | Pass `--setup-rag` (or `--setup-rag-ft`) — old vectors don't match.  |
| `BadStatusCode 404` from Qdrant on first batch run     | `--setup-*` first; the collection must exist before `--batch-*`.     |
| `Sentence Transformers version 5.3.0 vs 5.1.0` warning | Harmless; the FT was saved with a newer ST. Outputs are unaffected.  |
| Same numbers as before after editing input JSON        | Pass `--setup-*` — the Qdrant collection is cached.                  |

---

## 12. Out-of-scope (today)

- **FAISS / Unsloth / Ollama-FT variants** — see `examples/eval2/predict_pub242_v2.py`
  for those (six methods total).
- **Cross-method correlation** (predict_pub242_v2's `run_correlate`). The two
  JSONs share a schema, so a downstream merge is straightforward.
- **Random sampling** for `-n`. `eval.py` has it; `eval2.py` doesn't (yet).
  Take the first N or write a one-off slicer.
- **Reranking, hybrid (BM25 + dense) search.**
- **Cross-corpus retrieval** (asking pub242 questions against techbriefs
  answers, or vice versa). That's `eval.py`'s job.
