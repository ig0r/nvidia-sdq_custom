# `eval.py` — pub242 vs techbriefs retrieval evaluation

Single-script evaluator that asks pub242 hq questions against an embedding-based
retrieval index and reports top-K matches under three pool-restriction modes:

| Mode               | Candidate pool                                       |
|--------------------|------------------------------------------------------|
| `pub242_only`      | only pub242 hq answers (≈9 296 docs)                 |
| `techbriefs_only`  | only techbriefs c-eval answers (≈40 137 docs)        |
| `both`             | unified pool (≈49 433 docs)                          |

Both corpora are embedded once into a **single Qdrant collection**; modes are
implemented as Qdrant `Filter` clauses on a `source` payload field — vectors are
bit-identical across modes.

Companion design docs:
- `plans/plan-eval-pub242-vs-techbriefs.md`
- `plans/srs-eval-pub242-vs-techbriefs.md`

---

## 1. What it does

For each pub242 hq question:

1. Embeds the question with `BAAI/bge-base-en-v1.5` (single model load, cached).
2. Issues a Qdrant `query_points` call with the mode's source filter, retrieving
   `top_k + 1` candidates so the *low-boundary* metric still has `top_k` items
   after self-match removal.
3. Computes three metric families per `k ∈ match_at_k` (default `[1, 3, 5]`):
   - **`qid_top{k}`** — query's own `question_id` is in top-K (self-retrieval).
   - **`proc_top{k}`** — query's `proc_id` (e.g. `6.1-0`) appears in any pub242
     hit in top-K.
   - **`proc_lb_top{k}`** — same as `proc`, but with the self-match prediction
     removed first. Tells you whether the model can find the right pavement
     process via *other* answers.
4. Saves predictions + match flags to one JSON + one CSV per mode.

Each saved prediction carries full provenance:

- pub242 hits: `proc_id`, `process_id`, `chapter_number`, `chapter_title`,
  `section_number`, `section_title`.
- techbriefs hits: `doc_id`, `u_ctx_id`, `u_logic_chunk_id`, `source_u_chunk_ids`,
  `artifact_id`, `u_artifact_id`.

This lets downstream analysis answer not just *"did it retrieve a hit?"* but
*"what content did it retrieve, in which logical chunk / context / chapter?"*.

---

## 2. Quick start

```bash
cd /Users/igor/dev/llm/pavement-gpt/nvidia-pipeline/nvidia-sdq_custom

# 1) Smoke test on first 20 queries — builds the DB if missing.
python eval.py -n 20

# 2) Full run — every pub242 hq question, in every mode listed in eval.toml.
python eval.py

# 3) Force a rebuild (after corpus or model changes).
python eval.py --rebuild
```

First run downloads `BAAI/bge-base-en-v1.5` (~430 MB) into
`~/.cache/huggingface/` and encodes ~49k passages. On CPU this takes several
minutes; on a single GPU it's seconds. **Subsequent runs skip the build
entirely** — the Qdrant collection persists on disk.

---

## 3. CLI

```
python eval.py [--cfg PATH] [--rebuild] [--mode MODE] [-n N] [--top-k K]
```

| Flag           | Default                       | Effect                                                              |
|----------------|-------------------------------|---------------------------------------------------------------------|
| `--cfg`        | `./eval.toml`                 | TOML config path. Relative to script dir if missing.                |
| `--rebuild`    | off                           | Drop and re-create the Qdrant collection.                           |
| `--mode`       | `all`                         | `pub242_only` / `techbriefs_only` / `both` / `all`.                 |
| `-n`           | `[queries] num_questions`     | Limit to N queries. Overrides the TOML when present.                |
| `--selection`  | `[queries] selection`         | `sequential` (first N) or `random` (uniform sample with seed).      |
| `--seed`       | `[queries] seed`              | RNG seed for `random` selection (ignored when `sequential`).        |
| `--top-k`      | `[eval] top_k`                | Override retrieval depth. `match_at_k` clamps to it.                |

`--mode all` runs every entry of `[eval] modes` in the TOML (default order:
`pub242_only`, `techbriefs_only`, `both`).

### Query selection: sequential vs random

Two ways to limit how many questions get asked:

- **`sequential`** — take the first `N` entries from the loaded JSON. Fast,
  deterministic, but biased toward whatever ordering the source file has
  (typically chapter 6 first for `pub242_hq.json`). Good for dev iteration on
  a fixed slice.
- **`random`** — uniformly sample `N` entries with `random.Random(seed).sample`.
  Reproducible across runs as long as `seed` is fixed. Use this when you want
  the limited eval to actually represent the full corpus rather than a head
  slice — e.g. for fast smoke comparisons across modes or models.

The selection happens **once** in `main()` before the per-mode loop, so all
modes see the same question set — apples-to-apples comparison.

The chosen selection is recorded in each output JSON under
`model.query_selection` (`{"selection": "...", "limit": N, "seed": ..., "num_selected": N}`)
so a downstream reader can confirm what was actually asked.

---

## 4. Configuration (`eval.toml`)

```toml
Title = "Pub242 vs Techbriefs Retrieval Eval"

[corpus]
pub242_json     = "./examples/db-for-mason-pub/pub242_hq.json"
techbriefs_json = "./data/nemo_briefs_20260429/qa-gen-cluster/generated-questions_wo_context-c-eval.json"

[queries]
input         = "./examples/db-for-mason-pub/pub242_hq.json"
num_questions = 100              # 0 or omitted = use all
selection     = "sequential"     # "sequential" | "random"
seed          = 42               # used only when selection = "random"

[embedding]
model      = "BAAI/bge-base-en-v1.5"
dim        = 768
batch_size = 64

[qdrant]
db_path    = "./data/nemo_briefs_20260429/eval/qdrant_db"
collection = "answers_combined"

[eval]
top_k       = 5
match_at_k  = [1, 3, 5]
output_dir  = "./data/nemo_briefs_20260429/eval"
modes       = ["pub242_only"]   # add "techbriefs_only", "both" to expand
```

| Key                          | Purpose                                                      |
|------------------------------|--------------------------------------------------------------|
| `[corpus] pub242_json`       | Source of pub242 answers + metadata (chapter/section).       |
| `[corpus] techbriefs_json`   | Source of techbriefs answers + provenance fields.            |
| `[queries] input`            | JSON list of pub242 hq questions to query with.              |
| `[queries] num_questions`    | Max queries per run. `0` (or omitted) = ask all loaded.      |
| `[queries] selection`        | `sequential` (first N) or `random` (seeded uniform sample).  |
| `[queries] seed`             | RNG seed used when `selection = "random"`.                   |
| `[embedding] model`          | SentenceTransformers model name or local path.               |
| `[embedding] dim`            | Vector dimension. Must match the model.                      |
| `[embedding] batch_size`     | Encoding batch size (one tqdm bar over all docs).            |
| `[qdrant] db_path`           | Local on-disk Qdrant store. Created on first run.            |
| `[qdrant] collection`        | Single collection holds both corpora.                        |
| `[eval] top_k`               | Predictions retained per query.                              |
| `[eval] match_at_k`          | K values for the match-at-K columns (always ≤ `top_k`).      |
| `[eval] output_dir`          | Destination of `eval_<mode>.{json,csv}`.                     |
| `[eval] modes`               | Default mode list when `--mode all` is used.                 |

All path keys resolve **relative to the config file's parent** if not absolute.

---

## 5. Outputs

### File layout (under `[eval] output_dir`)

```
output_dir/
├── qdrant_db/                          # local Qdrant store (also under [qdrant] db_path)
├── eval_pub242_only.json
├── eval_pub242_only.csv
├── eval_techbriefs_only.json
├── eval_techbriefs_only.csv
├── eval_both.json
└── eval_both.csv
```

### JSON shape (per mode)

```json
{
  "model": {
    "embedding_model": "BAAI/bge-base-en-v1.5",
    "embedding_dim":   768,
    "mode":            "pub242_only",
    "collection":      "answers_combined",
    "top_k":           5,
    "match_at_k":      [1, 3, 5],
    "num_queries":     9296
  },
  "mode": "pub242_only",
  "results": [
    {
      "question_id":            "publication_242-..._section_6.1-proc-0-q-6",
      "question":               "When can one design be used for multiple ...",
      "ground_truth_qid":       "publication_242-..._section_6.1-proc-0-q-6",
      "ground_truth_proc":      "6.1-0",
      "ground_truth_proc_full": "publication_242-..._section_6.1-proc-0",
      "predictions": [
        {
          "rank":          1,
          "score":         0.93,
          "source":        "pub242",
          "question_id":   "...",
          "answer_snippet": "first 200 chars ...",
          "proc_id":       "6.1-0",
          "process_id":    "...",
          "chapter_number":"6", "chapter_title":"...",
          "section_number":"6.1", "section_title":"..."
        },
        { "rank": 2, "source": "techbriefs", "doc_id": "TBF000001_UKN000",
          "u_ctx_id": "...", "u_logic_chunk_id": "...", "...": "..." }
      ],
      "matches": {
        "qid":     {"top1": true,  "top3": true,  "top5": true},
        "proc":    {"top1": true,  "top3": true,  "top5": true},
        "proc_lb": {"top1": false, "top3": true,  "top5": true}
      }
    }
  ]
}
```

In `pub242_only` mode every prediction has `source == "pub242"`; in
`techbriefs_only` every prediction has `source == "techbriefs"` and all match
flags are `false` (no pub242 record can possibly be retrieved). In `both` mode
predictions are interleaved by score.

### CSV shape (one row per query)

```
question_id, question, ground_truth_proc,
pred_1_source, pred_1_qid, pred_1_proc_or_doc, pred_1_section_or_logic_chunk, score_1,
... ×top_k ...,
qid_top1_match,    qid_top3_match,    qid_top5_match,
proc_top1_match,   proc_top3_match,   proc_top5_match,
proc_lb_top1_match, proc_lb_top3_match, proc_lb_top5_match
```

`pred_i_proc_or_doc` is the pub242 `proc_id` for pub242 hits, `doc_id` for
techbriefs hits. `pred_i_section_or_logic_chunk` is `section_number` for pub242
hits, `u_logic_chunk_id` for techbriefs hits. Missing predictions are padded
with empty fields.

### Console summary (per mode)

```
[pub242_only] 9296 queries
  Top-1  qid: 8217/9296  proc: 8403/9296  proc_lb: 1842/9296
  Top-3  qid: 8954/9296  proc: 9012/9296  proc_lb: 3211/9296
  Top-5  qid: 9081/9296  proc: 9112/9296  proc_lb: 4017/9296
Output JSON: data/nemo_briefs_20260429/eval/eval_pub242_only.json
Output CSV:  data/nemo_briefs_20260429/eval/eval_pub242_only.csv
```

(numbers above are illustrative — your actual values will depend on the model.)

---

## 6. How modes work internally

A single Qdrant collection is built once. Each point's payload includes a
`source ∈ {"pub242", "techbriefs"}` field. At query time:

| Mode               | Filter                                                           |
|--------------------|------------------------------------------------------------------|
| `pub242_only`      | `Filter(must=[FieldCondition(key="source", match="pub242")])`    |
| `techbriefs_only`  | `Filter(must=[FieldCondition(key="source", match="techbriefs")])`|
| `both`             | `None` (no filter)                                               |

So switching modes does **not** require a rebuild — vectors are identical, only
the post-filter differs. This isolates the variable under test (pool
composition) from confounders (encoding, normalization, distance metric).

---

## 7. Idempotency

`ensure_db` checks `client.collection_exists(...)` and `points_count > 0`. If
both are true and `--rebuild` is not set, it logs:

```
[ensure_db] DB ready: collection='answers_combined', points=49433, dim=768
```

…and returns immediately. No re-encoding, no upsert.

The Qdrant store at `[qdrant] db_path` is the canonical state — delete that
directory (or pass `--rebuild`) to force a fresh build. The output JSON / CSV
files are overwritten on each run.

---

## 8. Sanity checks

After a smoke run, verify:

```bash
# Mode purity
jq '.results[].predictions[].source' output/eval_pub242_only.json | sort -u
# → "pub242" only

jq '.results[].predictions[].source' output/eval_techbriefs_only.json | sort -u
# → "techbriefs" only

# Combined-mode usually mixes both
jq '.results[0].predictions[].source' output/eval_both.json | sort -u
# → typically both

# Row alignment
awk -F, '{print NF}' output/eval_pub242_only.csv | sort -u
# → exactly one value (header width == row width)

# Cross-check pub242_only against the v2 baseline
diff <(jq '.results | length' output/eval_pub242_only.json) \
     <(jq '.results | length' examples/eval2/output_v2/pub242_rag_results.json)
```

`pub242_only.qid_top1` should match the `[rag]` numbers in
`examples/eval2/output_v2/pub242_rag_results.json` (same model, same answers,
same K).

---

## 9. Common operations

### Run a single mode without editing TOML

```bash
python eval.py --mode techbriefs_only -n 50
```

### Random sample of N questions across the corpus

```bash
# 200 random pub242 questions, reproducible with seed 42
python eval.py -n 200 --selection random --seed 42

# Same 200 questions across modes (selection happens once before the mode loop)
python eval.py -n 200 --selection random --seed 42 --mode all
```

### Compare two embedding models on the *same* random slice

Pin `seed` and `num_questions` in `eval.toml`, then:

```bash
python eval.py --rebuild                                # baseline model
# edit [embedding] model in eval.toml
python eval.py --rebuild                                # candidate model
# both runs hit the same 100 random questions; outputs end up in the same dir,
# so rename them between runs if you want both kept side-by-side.
```

### Try a deeper retrieval

```bash
python eval.py --top-k 20            # top-20; match_at_k auto-clamps
```

### Switch corpora

Edit `[corpus] pub242_json` or `[corpus] techbriefs_json` in `eval.toml`, then:

```bash
python eval.py --rebuild
```

### Switch embedding model

Edit `[embedding] model` and `[embedding] dim`. **Always pass `--rebuild`** —
vectors from the old model are now incompatible.

### Inspect what got retrieved for a particular question

```bash
jq '.results[] | select(.question_id == "publication_242-...-q-6") | .predictions' \
   data/nemo_briefs_20260429/eval/eval_both.json
```

---

## 10. Metrics — what they tell you

| Metric             | Reads as                                                              |
|--------------------|-----------------------------------------------------------------------|
| `qid_top{k}`       | "Did the embedder find the original answer to this question?"        |
| `proc_top{k}`      | "Did it find *any* answer that belongs to the right pavement process?"|
| `proc_lb_top{k}`   | "If we *hide* the original answer, does the embedder still surface a sibling answer from the same process?" |

`proc_lb` is the most informative metric for pub242 because the corpus is built
from the same hq questions used to train embeddings — `qid_top1` is therefore
trivially high in `pub242_only` mode (training-set overlap). `proc_lb` shows
whether the embedder generalizes from one Q&A in a process to its siblings.

In `techbriefs_only` mode all three families read as zero by construction
(no pub242 records are in the pool). The value of that mode is in the
**predictions list** — which techbriefs answers come up for each pub242
question, and which logical chunk / context / artifact they point to.

---

## 11. Troubleshooting

| Symptom                                            | Fix                                              |
|----------------------------------------------------|--------------------------------------------------|
| `Error: config file not found: ./eval.toml`       | `--cfg <path>` or run from script dir.           |
| `Error: missing config keys: [corpus] pub242_json`| Add the missing key to your TOML.                |
| `Error: invalid modes in config: ['foo']`         | Use only `pub242_only` / `techbriefs_only` / `both`. |
| `ModuleNotFoundError: qdrant_client`              | `pip install -r reqs.txt` in your venv.          |
| `[pub242] metadata parse failed: ...` (stderr)    | A row's `process_source_metadata` couldn't be parsed; identity fields are kept, chapter/section enrichment is dropped for that row. Investigate the offending row if many fire. |
| Same numbers as before after corpus change         | Pass `--rebuild` — the Qdrant store is cached.   |
| Different `dim` / shape error after model swap    | Pass `--rebuild` after editing `[embedding]`.    |

---

## 12. Out-of-scope (today)

- FAISS, Ollama, Unsloth, finetuned variants — see `examples/eval2/predict_pub242_v2.py` for those.
- Cross-mode correlation (v2's `run_correlate`). The three JSONs share schema, so a downstream merge is straightforward.
- Reranking, hybrid (BM25 + dense) search.
- Asymmetric query/document prompts. `bge-base-en-v1.5` is symmetric, so the
  `hasattr(model, "encode_query")` guard is a no-op today; swapping in
  `Qwen3-Embedding-0.6B` would activate the asymmetric path.
