# Plan: pub242 vs techbriefs retrieval eval (`eval.py`)

## Context

`examples/eval2/predict_pub242_v2.py` evaluates self-retrieval over pub242's hq Q&A: it embeds the answers from `pub242_hq.json`, queries with pub242 questions, and reports top-K self-match metrics across multiple backends (Ollama / SentenceTransformers / Unsloth) and two retrievers (Qdrant / FAISS).

We now need a **second-corpus** evaluation: when answering pub242 hq questions, how does retrieval perform when the candidate pool is (a) only pub242 answers, (b) only techbriefs answers, (c) the union? This tells us how much of pub242's hq Q&A is already covered by techbriefs content (`data/nemo_briefs_20260429/qa-gen-cluster/generated-questions_wo_context-c-eval.json`, 40 137 records) and where the gaps are. The new script lives at the repo root and is intentionally simpler than v2 — single embedding model, single retriever, no FAISS / CSV / Ollama / Unsloth variants.

## Scope

**In scope**

- New script `eval.py` at repo root.
- New config `eval.toml` at repo root.
- One Qdrant collection (`answers_combined`) holding both corpora; mode selection is a payload filter, not a separate index.
- Three eval modes per run, each emitting one JSON + one CSV under `./output/`:
  - `pub242_only` — query pool restricted to pub242 answers (mirrors v2 `[rag]`).
  - `techbriefs_only` — query pool restricted to techbriefs answers.
  - `both` — unified pool, single ranking.
- Metrics: `qid_top{1,3,5}`, `proc_top{1,3,5}`, `proc_lb_top{1,3,5}` (same families as v2 — kept for schema consistency; for `techbriefs_only` they are 0 by construction).
- Companion SRS `plans/srs-eval-pub242-vs-techbriefs.md`.

**Out of scope**

- FAISS retriever, Ollama backend, finetuned / Unsloth variants. Single backend = SentenceTransformers + Qdrant.
- Cross-mode correlation analysis (v2's `run_correlate`). The three JSONs share schema, so it can be bolted on later.
- CSV import/export of the corpus (the v2 `simple-rag` `docs_csv` is not produced).
- Embedding-model swap CLI flag — model is read from TOML only.
- Reranking. Pure cosine on bge-base-en-v1.5 vectors.
- A "limit" knob for techbriefs corpus size — full ~40k pool is required (user requirement #2).

## Concrete changes

### New: `eval.py`

Top-level standalone script. Layout (source order):

1. **Imports + tomllib/tomli shim.**
2. **Config:** `load_config(path) -> dict` — `tomllib.load` then resolve relative paths under `[corpus]`, `[queries]`, `[qdrant]`, `[eval]` against the config file's parent (mirrors `examples/eval2/predict_pub242_v2.py:64-103`).
3. **Helpers — copied verbatim from v2** (no cross-file import, keeps the script standalone):
   - `_get_st_model(path)` — `examples/eval2/predict_pub242_v2.py:177-181`.
   - `embed_st_query(text, model_path)` — v2:209-213, keeps `hasattr(model, "encode_query")` fallback.
   - `_encode_documents_batched(model, texts, batch_size)` — v2:223-239.
   - `embed_st_documents_batch(texts, model_path, batch_size)` — v2:242-245.
   - `extract_proc_id(process_id)` — v2:257-265.
   - `_answer_snippet(text, n=200)` — v2:308-312.
   - `_upsert_batched(client, collection, points, batch=500)` — v2:530-532.
4. **Doc loading:**
   - `_parse_pub242_metadata(s) -> dict` — try `ast.literal_eval`, then `json.loads`, on failure log a stderr WARNING and return `{}` so a single bad row doesn't abort the build.
   - `_load_pub242_docs(path) -> list[dict]` — drops empty-answer rows; payload includes `source="pub242"`, `question_id`, `answer`, `process_id`, `proc_id` (via `extract_proc_id`), `source_chunk_id`, plus parsed metadata fields (`chapter_number`, `chapter_title`, `section_number`, `section_title`, `source_id`, `source_title`).
   - `_load_techbriefs_docs(path) -> list[dict]` — drops empty-answer rows; payload includes `source="techbriefs"`, `question_id`, `answer`, `doc_id`, `u_ctx_id`, `u_logic_chunk_id`, `source_u_chunk_ids` (list), `artifact_id`, `u_artifact_id`, `artifact_description` (from `artifact.description`).
5. **DB:**
   - `_build_db(client, cfg)` — load both corpora, drop+recreate collection with `VectorParams(size=cfg.embedding.dim, distance=COSINE)`, encode all answers in one `embed_st_documents_batch` call (one tqdm bar, one model load), build `PointStruct(id=uuid.uuid5(NAMESPACE_DNS, f"{source}:{question_id}"), vector=..., payload=full_doc_dict)`, upsert in 500-batches.
   - `ensure_db(cfg, force=False) -> QdrantClient` — same idempotency logic as v2's `ensure_rag_db` (v2:535-561): mkdir db_path → `QdrantClient(path=db_path)` → check `collection_exists` and `points_count > 0` → rebuild if forced/missing/empty, else log "DB ready" and return.
6. **Retrieval:**
   - `_make_filter(mode) -> Optional[Filter]` — `Filter(must=[FieldCondition(key="source", match=MatchValue(value=...))])` for the two single-source modes, `None` for `both`.
   - `_retrieve(client, collection, qvec, k, mode) -> list[pred_dict]` — `client.query_points(..., query_filter=_make_filter(mode), limit=k, with_payload=True)`.
   - `_normalize_pred(rank, hit) -> dict` — flat record with `rank`, `score`, `source`, `question_id`, `answer_snippet`, plus source-specific fields (pub242 → proc/process/chapter/section; techbriefs → doc/ctx/logic-chunk/artifact). Source-irrelevant fields are simply absent (not null).
7. **Eval:** `_eval_one_mode(cfg, client, queries, mode)` — per-mode loop following v2's `_run_eval` (v2:333-462). Computes `qid` / `proc` / `proc_lb` matches, accumulates counters, writes one JSON + one CSV.
8. **Output writers:**
   - `_write_json(path, model_info, mode, results)` — `{"model": {...}, "mode": "...", "results": [...]}`.
   - `_write_csv(path, results, top_k, match_at_k)` — flat row per query (header in §"CSV format" below). Padding rule: missing predictions → `["", "", "", "", ""]` per slot.
9. **CLI** (`main()`): `--cfg`, `--rebuild`, `--mode`, `-n`, `--top-k`. Default `--mode=all` runs every entry in `cfg.eval.modes`.

### New: `eval.toml`

```toml
[corpus]
pub242_json     = "./examples/db-for-mason-pub/pub242_hq.json"
techbriefs_json = "./data/nemo_briefs_20260429/qa-gen-cluster/generated-questions_wo_context-c-eval.json"

[queries]
input = "./examples/db-for-mason-pub/pub242_hq.json"

[embedding]
model      = "BAAI/bge-base-en-v1.5"
dim        = 768
batch_size = 64

[qdrant]
db_path    = "./output/qdrant_db"
collection = "answers_combined"

[eval]
top_k       = 5
match_at_k  = [1, 3, 5]
output_dir  = "./output"
modes       = ["pub242_only", "techbriefs_only", "both"]
```

### New: `plans/srs-eval-pub242-vs-techbriefs.md`

Companion SRS — see that file.

### Output files (under `cfg.eval.output_dir`, default `./output/`)

| Mode             | JSON                            | CSV                            |
|------------------|---------------------------------|--------------------------------|
| pub242_only      | `eval_pub242_only.json`         | `eval_pub242_only.csv`         |
| techbriefs_only  | `eval_techbriefs_only.json`     | `eval_techbriefs_only.csv`     |
| both             | `eval_both.json`                | `eval_both.csv`                |

JSON top-level: `{"model": {...}, "mode": "...", "results": [...]}`. Each result row carries the full normalized prediction list including source-specific provenance fields, so a downstream techbriefs hit shows its `u_logic_chunk_id`, `u_ctx_id`, `doc_id`, `artifact_id`.

### CSV format (one row per query)

```
question_id, question, ground_truth_proc,
pred_1_source, pred_1_qid, pred_1_proc_or_doc,
pred_1_section_or_logic_chunk, score_1,
pred_2_source, pred_2_qid, pred_2_proc_or_doc,
pred_2_section_or_logic_chunk, score_2,
... ×top_k ...,
qid_top1_match, qid_top3_match, qid_top5_match,
proc_top1_match, proc_top3_match, proc_top5_match,
proc_lb_top1_match, proc_lb_top3_match, proc_lb_top5_match
```

`pred_i_proc_or_doc` = `proc_id` for pub242 hits, `doc_id` for techbriefs hits.
`pred_i_section_or_logic_chunk` = `section_number` for pub242 hits, `u_logic_chunk_id` for techbriefs hits.

### Unchanged

- `examples/eval2/predict_pub242_v2.py`, `examples/eval2/predict_pub242_v2.toml` — reference only, not edited.
- `_nemo.py`, `aisa/`, `cfg/nemo.toml`, prompts — untouched.
- `reqs.txt` — no new dependencies (`qdrant-client`, `sentence-transformers`, `tqdm`, `tomli`/`tomllib`, `numpy` — all required by the v2 script already).

## Behavior notes

- **Single collection, payload filter for modes.** The corpus is embedded once. `pub242_only`, `techbriefs_only`, `both` differ only in the Qdrant `Filter` applied at query time; vectors are bit-identical across modes. This guarantees that any quality difference between modes reflects pool composition, not embedding drift.
- **Symmetric model, symmetric calls.** `BAAI/bge-base-en-v1.5` is a symmetric encoder — `encode_query` and `encode_document` collapse to plain `encode`. The `hasattr(model, "encode_query")` guard is preserved verbatim from v2 to keep the door open for swapping in an asymmetric model (e.g. Qwen3-Embedding) later.
- **Idempotent build.** The collection is rebuilt only when missing, empty, or `--rebuild` is passed. A re-run after a successful build skips encoding entirely and goes straight to query.
- **Stable point IDs across sources.** `uuid5(NAMESPACE_DNS, f"{source}:{question_id}")` namespaces by source so a hypothetical `question_id` collision between pub242 and techbriefs (none observed empirically) cannot overwrite either point.
- **Robust pub242 metadata parsing.** `process_source_metadata` is a Python-literal string with single quotes (`{'id': '...', ...}`). `ast.literal_eval` handles this directly; `json.loads` is a fallback for any future change to JSON-quoted strings; on failure the row keeps its core pub242 identity (qid, proc_id, process_id) and just loses the chapter/section enrichment.
- **`techbriefs_only` matches columns are always 0.** No pub242 `question_id` or `proc_id` can appear in a techbriefs-only pool. The columns are still emitted to keep one CSV/JSON schema across modes; the value of this mode is in the prediction list itself, not the match flags.
- **`proc_lb` (low-boundary).** Same as v2: removes the self-match prediction (`question_id == query.question_id`) before computing `proc_top{k}`. Highlights whether the model can find the right pavement process via *other* answers, not just the trivially identical one.
- **No re-encoding when `top-k` changes.** `--top-k` only affects retrieval, not the DB.
- **Path resolution.** All `[corpus] / [queries] / [qdrant] / [eval]` path keys are resolved relative to the config file's parent if not absolute. Running `python eval.py` from the repo root with the default config (sibling of the script) produces `./output/...` next to the repo root.

## Verification

End-to-end smoke run (small `-n`, all modes, no rebuild):

```bash
cd /Users/igor/dev/llm/pavement-gpt/nvidia-pipeline/nvidia-sdq_custom
python eval.py -n 20
```

Expected:
- First run: ~49 433 points (~9 296 pub242 with non-empty answers + ~40 137 techbriefs) embedded with bge-base; one tqdm bar, then three eval bars.
- Second run (no `--rebuild`): `[ensure_db] DB ready: collection='answers_combined', points=49xxx, dim=768`; build skipped.
- Three JSON + three CSV files under `./output/`.

Cross-checks:

- **`pub242_only` ≈ v2 baseline.** Same model (`BAAI/bge-base-en-v1.5`), same answers, same metric → `pub242_only` `qid_top1` should match v2's `[rag]` numbers in `examples/eval2/output_v2/pub242_rag_results.json` to within rounding (identical vectors + cosine + same K).
- **Mode-purity check.** `jq '.results[].predictions[].source' output/eval_pub242_only.json | sort -u` → `"pub242"` only. Analogous for techbriefs.
- **Union-mode mix.** `jq '.results[0].predictions[].source' output/eval_both.json` → typically a mix; techbriefs density (~4.3× pub242) often dominates for short factual queries.
- **Metadata parse robustness.** Grep stderr for `[pub242] metadata parse failed`. Expected: zero on the current `pub242_hq.json`; the guard exists for future inputs.
- **CSV row alignment.** `awk -F, '{print NF}' output/eval_pub242_only.csv | sort -u` → exactly two values (header row width = data row width).

Spot-checks per mode (one query each):

- Pick a query with a known answer in pub242. In `pub242_only`, expect rank-1 to be the self-match (qid identical). In `techbriefs_only`, expect a thematically related techbrief in rank-1 (often a TBF on the same topic). In `both`, the self-match should still rank-1 most of the time.
- Pick a pub242 query that the v2 baseline misses. In `techbriefs_only` and `both`, inspect whether techbriefs covers it.

## Critical files referenced

- `examples/eval2/predict_pub242_v2.py:64-103` — `load_config` shape to mirror.
- `examples/eval2/predict_pub242_v2.py:177-245` — sentence-transformers helpers to copy.
- `examples/eval2/predict_pub242_v2.py:257-312` — `extract_proc_id`, `_answer_snippet`.
- `examples/eval2/predict_pub242_v2.py:333-462` — `_run_eval` shape (counters, JSON+CSV writers, tqdm postfix).
- `examples/eval2/predict_pub242_v2.py:530-561` — `_upsert_batched`, `ensure_rag_db` idempotency logic.
- `examples/eval2/predict_pub242_v2.toml` — TOML layout reference.
- `examples/db-for-mason-pub/pub242_hq.json` — corpus + queries source.
- `data/nemo_briefs_20260429/qa-gen-cluster/generated-questions_wo_context-c-eval.json` — second corpus.
- `plans/plan-filter-questions-citation-eval.md`, `plans/srs-filter-questions-citation-eval.md` — house-style reference for these plan/SRS docs.

## Decisions flagged

- **Single Qdrant collection vs two collections.** Single collection + payload filter is operationally simpler (one build, one disk path, one model load) and guarantees identical vectors across modes. Two collections would let us tune `dim` per source or use different embedders, but we don't need either.
- **Skip FAISS / Ollama / Unsloth.** v2 already covers those. The new question is "what does the candidate pool look like?", not "what backend?". Adding backends multiplies output files for no benefit.
- **`match_at_k = [1, 3, 5]` not `[1..5]`.** User requested top1/top3/top5 explicitly. Cuts the CSV column count and matches the natural reporting buckets.
- **Pub242 metadata: `ast.literal_eval` first, `json.loads` second.** The current data uses Python-literal repr (single quotes), which `json.loads` rejects. `ast.literal_eval` is safe (only literals, no code execution). The JSON fallback is a hedge for future inputs.
- **Helpers copied, not imported.** Keeps `eval.py` standalone and matches `filter-questions-citation-eval.py`'s convention. The duplication is small (≈80 lines) and makes the script grep-friendly.
- **Preserve v2's `proc_lb` metric.** Even though `pub242_only` is a self-retrieval setup where rank-1 is almost always the self-match, the low-boundary view ("can we find the right process via *other* answers?") is still informative and the cost is one extra `[:top_k]` slice.
- **No reranking, no hybrid search.** Out of scope; can be added once we know the baseline numbers.
