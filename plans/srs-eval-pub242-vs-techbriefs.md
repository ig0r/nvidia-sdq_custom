# Software Requirements Specification: pub242 vs techbriefs retrieval eval

**Feature:** Add a standalone evaluation script that embeds two answer corpora (pub242 hq Q&A + techbriefs c-eval Q&A) into a single Qdrant collection with a `source` payload tag, queries the collection with pub242 hq questions under three pool-restriction modes (`pub242_only`, `techbriefs_only`, `both`), and writes per-mode JSON + CSV reports of top-K predictions and self-retrieval / proc-co-membership metrics. Embedding is done once with `BAAI/bge-base-en-v1.5`; modes differ only in a Qdrant `Filter` applied at query time, so vectors are bit-identical across modes.

**Component:** `nvidia-sdq_custom`
**Version:** 0.1 (draft)
**Status:** Proposed
**Companion plan:** `plans/plan-eval-pub242-vs-techbriefs.md`

---

## 1. Introduction

### 1.1 Purpose

This SRS defines requirements for a small, self-contained Python script that evaluates how a retrieval system over pub242's hq questions performs depending on which answer corpus is exposed to the retriever:

- pub242 answers only (mirrors the v2 `[rag]` baseline at `examples/eval2/predict_pub242_v2.py`),
- techbriefs answers only (does pub242's content overlap with the techbriefs corpus?),
- the union (does adding techbriefs help or hurt pub242 self-retrieval?).

The script is the second-corpus generalization of `examples/eval2/predict_pub242_v2.py`. It targets a single embedding model and a single retriever (Qdrant), so the comparison axis is *pool composition*, not backend.

### 1.2 Scope

In scope:
- A new script `eval.py` at the repo root.
- A new TOML config `eval.toml` at the repo root.
- A single Qdrant collection (`answers_combined`) holding both corpora, distinguished by a `source` payload field.
- Three eval modes per run (selectable via `--mode` and `[eval] modes`).
- Two output files per mode (JSON + CSV) under `[eval] output_dir`.
- A companion plan `plans/plan-eval-pub242-vs-techbriefs.md` and this SRS.

Out of scope:
- FAISS / Ollama / finetuned / Unsloth backends from v2.
- Cross-mode correlation analysis (v2's `run_correlate`).
- Reranking, hybrid search, query expansion.
- Embedding-model swap via CLI flag.
- Limit knob for the techbriefs corpus (the user requires the full pool).

### 1.3 Definitions

- **pub242 hq record** — one element of `examples/db-for-mason-pub/pub242_hq.json`. Dict with at least `question_id, question, answer, process_id, process_source_metadata, source_chunk_id`.
- **techbriefs c-eval record** — one element of `data/nemo_briefs_20260429/qa-gen-cluster/generated-questions_wo_context-c-eval.json`. Dict with at least `question_id, question, answer, doc_id, u_ctx_id, u_logic_chunk_id, source_u_chunk_ids, artifact_id, u_artifact_id, artifact`.
- **Doc record** — payload-ready dict produced by `_load_pub242_docs` / `_load_techbriefs_docs`. Always carries `source ∈ {"pub242", "techbriefs"}`, `question_id`, `answer`, plus source-specific provenance fields (see FR-3.5, FR-3.6).
- **`process_source_metadata`** — a Python-literal *string* on each pub242 row (single-quoted dict). Parsed via `ast.literal_eval` with `json.loads` fallback.
- **`proc_id`** — short canonical form of `process_id` produced by `extract_proc_id` (e.g. `publication_242-..._section_6.1-proc-0` → `6.1-0`).
- **Mode** — one of `pub242_only`, `techbriefs_only`, `both`. Maps to a Qdrant `Filter` (or `None` for `both`).
- **`top_k`** — number of predictions retained per query (default 5).
- **`match_at_k`** — list of K values at which match flags are computed (default `[1, 3, 5]`).
- **qid match @ K** — query's own `question_id` appears in the top-K predictions.
- **proc match @ K** — query's `proc_id` appears in the top-K predictions (via any pub242 hit; techbriefs hits cannot match by definition).
- **proc_lb (low-boundary) @ K** — `proc` match at K computed after removing the self-`question_id` prediction; reflects whether the right pavement process is reachable via *other* answers.
- **Self-retrieval** — querying with one record's `question` and finding the same record's answer in the result list (`question_id` identical).

### 1.4 References

- `plans/plan-eval-pub242-vs-techbriefs.md` — companion implementation plan.
- `examples/eval2/predict_pub242_v2.py:64-103` — `load_config` and path-resolution shape.
- `examples/eval2/predict_pub242_v2.py:177-245` — sentence-transformers helpers (`_get_st_model`, `embed_st_query`, `_encode_documents_batched`, `embed_st_documents_batch`).
- `examples/eval2/predict_pub242_v2.py:257-265` — `extract_proc_id`.
- `examples/eval2/predict_pub242_v2.py:308-312` — `_answer_snippet`.
- `examples/eval2/predict_pub242_v2.py:333-462` — `_run_eval` shape (counters, JSON+CSV writers, tqdm postfix).
- `examples/eval2/predict_pub242_v2.py:530-561` — `_upsert_batched`, `ensure_rag_db` (idempotency).
- `examples/eval2/predict_pub242_v2.toml` — TOML layout reference.
- `examples/db-for-mason-pub/pub242_hq.json` — corpus + queries source.
- `data/nemo_briefs_20260429/qa-gen-cluster/generated-questions_wo_context-c-eval.json` — second corpus.
- `plans/plan-filter-questions-citation-eval.md`, `plans/srs-filter-questions-citation-eval.md` — house-style reference for these plan/SRS docs.

---

## 2. Overall Description

### 2.1 Product Perspective

`predict_pub242_v2.py` answers "how well does a baseline / finetuned embedder do at self-retrieval over pub242?". The new script answers a different question: **given a fixed embedder, how much of pub242's hq Q&A is already discoverable via techbriefs answers, and does mixing the two corpora help or hurt pub242's own self-retrieval?**

The flow is:

1. Load both corpora into payload-ready doc records (with `source` tag).
2. Encode all answers once with `BAAI/bge-base-en-v1.5` and upsert into a single Qdrant collection.
3. For each pub242 hq question, run three filtered retrievals (one per mode) and emit predictions + metrics.

The script is otherwise structurally a slimmed-down v2: same metrics families, same JSON/CSV shape, same idempotency pattern. The novelty is the dual-corpus payload + the per-mode Qdrant filter.

### 2.2 User Classes

- **Pipeline operator** — runs `python eval.py` to compare retrieval quality across the three pool modes. Edits `[corpus]` paths in `eval.toml` if input locations change. Reads JSON / CSV outputs and the console summary.
- **Pipeline developer** — extends the script (e.g. add reranking, swap embedder, add a fourth mode that filters by `chapter_number`). Updates this SRS when behavior changes.
- **Downstream consumer** — analyzes `output/eval_*.json` to surface gaps where pub242 hq questions are not covered by either pool.

### 2.3 Operating Environment

- Python 3.11+ (uses `tomllib`; `tomli` shim included for older interpreters).
- Dependencies, all already required by `predict_pub242_v2.py` and `reqs.txt`: `qdrant-client`, `sentence-transformers`, `numpy`, `tqdm`, `tomllib`/`tomli`. No `aisa.*` imports, no LLM credentials, no Ollama, no FAISS.
- Disk: `~50k × 768-dim float32` ≈ 150 MB for the Qdrant collection.
- GPU optional. Encoding ~50k passages on CPU takes minutes; on a single GPU, seconds.
- The script runs from any CWD; relative paths in TOML resolve against the config file's parent.

### 2.4 Constraints

- The script SHALL only read its config file, the two corpus JSON files, the queries JSON file, and (after build) the local Qdrant DB directory.
- The script SHALL only write the Qdrant DB directory and the JSON/CSV outputs under `output_dir`.
- The script SHALL NOT modify any input file.
- The script SHALL build the collection only if it is missing or empty, unless `--rebuild` is passed.
- Modes SHALL be expressed as Qdrant `Filter` clauses on the `source` payload field; the script SHALL NOT maintain separate collections per mode.
- The embedding model SHALL be loaded at most once per process (cached in `_st_cache`).
- The script SHALL be synchronous. No `asyncio`, no threads (the underlying SentenceTransformer batch path is the parallelism boundary).

### 2.5 Assumptions

- `pub242_hq.json` is a top-level JSON list and each row is a dict with the fields named in §1.3.
- `generated-questions_wo_context-c-eval.json` is a top-level JSON list and each row is a dict with the fields named in §1.3.
- `process_source_metadata` is parsable as a Python literal in ≥99% of pub242 rows; rows where it fails to parse retain their identity fields and lose only the chapter/section enrichment.
- `BAAI/bge-base-en-v1.5` is downloadable on first run (or already cached in `~/.cache/huggingface/`).
- Qdrant local-disk mode (`QdrantClient(path=...)`) is sufficient for a single-process eval (no concurrent writers).
- The operator runs the script from a path where `./output/` is writable.

---

## 3. Functional Requirements

### FR-1 — Config loading

- **FR-1.1** The script SHALL accept `--cfg <path>` (default `<script_dir>/eval.toml`), `--rebuild` (flag), `--mode {pub242_only,techbriefs_only,both,all}` (default `all`), `-n <int>` (limit queries), `--top-k <int>` (override `[eval] top_k`).
- **FR-1.2** The script SHALL load the TOML config via `tomllib.load`, mirroring `examples/eval2/predict_pub242_v2.py:64-66`.
- **FR-1.3** The script SHALL resolve every value under `[corpus]`, `[queries]`, `[qdrant]`, `[eval]` whose key is one of `{pub242_json, techbriefs_json, input, db_path, output_dir}` against the config file's parent if not absolute. All other keys are passed through as-is.
- **FR-1.4** Missing required keys (`[corpus] pub242_json`, `[corpus] techbriefs_json`, `[queries] input`, `[embedding] model`, `[embedding] dim`, `[qdrant] db_path`, `[qdrant] collection`) SHALL cause an ERROR-level message and a non-zero exit.
- **FR-1.5** `[eval] modes` SHALL default to `["pub242_only", "techbriefs_only", "both"]` if absent. Each entry MUST be one of those three strings; otherwise non-zero exit.
- **FR-1.6** `--mode all` SHALL run every entry of `cfg.eval.modes` in order. `--mode <name>` SHALL run only that mode (and SHALL fail non-zero if `<name>` is not a valid mode).

### FR-2 — Corpus loading

- **FR-2.1** The script SHALL define `_parse_pub242_metadata(s: str) -> dict`:
  1. If `s` is empty / not a string → return `{}`.
  2. Try `ast.literal_eval(s)`; if result is a `dict` → return it.
  3. On `ValueError` / `SyntaxError`, try `json.loads(s)`; if result is a `dict` → return it.
  4. On any failure, log `[pub242] metadata parse failed: <first 80 chars>...` to stderr at WARNING and return `{}`.
- **FR-2.2** `_load_pub242_docs(path) -> list[dict]` SHALL read the JSON list, drop rows where `answer.strip()` is empty, and return doc records with the keys named in §1.3 (Doc record).
  - SHALL set `source = "pub242"`.
  - SHALL set `proc_id = extract_proc_id(process_id)` per the v2 helper (`examples/eval2/predict_pub242_v2.py:257-265`).
  - SHALL pull `chapter_number, chapter_title, section_number, section_title, source_id` from the parsed metadata; missing fields fall back to `""`.
  - SHALL set `source_title = md.get("source_title_short") or md.get("source_title", "")`.
- **FR-2.3** `_load_techbriefs_docs(path) -> list[dict]` SHALL read the JSON list, drop rows where `answer.strip()` is empty, and return doc records with:
  - `source = "techbriefs"`, `question_id`, `answer`.
  - `doc_id, u_ctx_id, u_logic_chunk_id, source_u_chunk_ids, artifact_id, u_artifact_id` from the row, defaulting to `""` (or `[]` for `source_u_chunk_ids`).
  - `artifact_description = (row.get("artifact") or {}).get("description", "")`.
- **FR-2.4** Both loaders SHALL log INFO `[load] pub242: kept N / total M` and `[load] techbriefs: kept N / total M`.

### FR-3 — Database build

- **FR-3.1** `ensure_db(cfg, force=False) -> QdrantClient` SHALL `Path(db_path).mkdir(parents=True, exist_ok=True)` and instantiate `QdrantClient(path=db_path)`.
- **FR-3.2** If `force` is False, the collection exists, and `client.get_collection(...).points_count > 0`, the function SHALL skip building and return the client. Otherwise it SHALL call `_build_db`.
- **FR-3.3** `_build_db(client, cfg)` SHALL:
  1. Concatenate `_load_pub242_docs(...)` + `_load_techbriefs_docs(...)`.
  2. If the collection exists, `client.delete_collection(collection)`.
  3. Create the collection with `VectorParams(size=cfg.embedding.dim, distance=Distance.COSINE)`.
  4. Encode all answers in one call: `embed_st_documents_batch([d["answer"] for d in docs], cfg.embedding.model, batch_size=cfg.embedding.batch_size)`.
  5. Build `PointStruct` records with `id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{d['source']}:{d['question_id']}"))`, `vector = emb.tolist()`, `payload = full_doc_dict`.
  6. Upsert via `_upsert_batched(client, collection, points, batch=500)` (copied from v2:530-532).
- **FR-3.4** Build SHALL log INFO `[build] N pub242 + M techbriefs = K total` and `[build] upserted K points to '<collection>'`.
- **FR-3.5** Embedding model SHALL be loaded via `_get_st_model` (copied from v2:177-181) so a subsequent query phase reuses the same instance.

### FR-4 — Query loop

- **FR-4.1** The script SHALL load queries from `cfg.queries.input` as a top-level JSON list. With `-n N`, only the first `N` are used.
- **FR-4.2** For each query, the script SHALL compute one query vector via `embed_st_query(question, cfg.embedding.model)` (copied from v2:209-213, retains the `hasattr(model, "encode_query")` fallback).
- **FR-4.3** The query vector SHALL be reused across modes (no re-encoding per mode).
- **FR-4.4** For each requested mode the script SHALL call `_retrieve(client, collection, qvec, top_k + 1, mode)`. The `+1` matches the v2 pattern so `proc_lb` still has `top_k` predictions after self-match removal.
- **FR-4.5** `_make_filter(mode)` SHALL return:
  - `Filter(must=[FieldCondition(key="source", match=MatchValue(value="pub242"))])` for `pub242_only`.
  - `Filter(must=[FieldCondition(key="source", match=MatchValue(value="techbriefs"))])` for `techbriefs_only`.
  - `None` for `both`.
- **FR-4.6** `_retrieve` SHALL call `client.query_points(collection_name=collection, query=qvec.tolist(), query_filter=_make_filter(mode), limit=k, with_payload=True).points` and return a list of dicts via `_normalize_pred(rank, hit)`.

### FR-5 — Prediction normalization

- **FR-5.1** `_normalize_pred(rank, hit)` SHALL emit a flat dict with: `rank, score (float), source, question_id, answer_snippet`.
- **FR-5.2** When `source == "pub242"`, the dict SHALL additionally include `proc_id, process_id, chapter_number, chapter_title, section_number, section_title`.
- **FR-5.3** When `source == "techbriefs"`, the dict SHALL additionally include `doc_id, u_ctx_id, u_logic_chunk_id, source_u_chunk_ids, artifact_id, u_artifact_id`.
- **FR-5.4** `answer_snippet` SHALL be the first 200 characters of the payload `answer` (via `_answer_snippet` from v2:308-312).

### FR-6 — Metrics

- **FR-6.1** The script SHALL compute per-query, for each `k in match_at_k`:
  - `qid_match[topK] = (query.question_id ∈ {p.question_id for p in preds[:k]})`.
  - `proc_match[topK] = (gt_proc != "" and gt_proc ∈ {p.get("proc_id", "") for p in preds[:k]})`.
  - `proc_lb_match[topK] = (gt_proc != "" and gt_proc ∈ {p.get("proc_id", "") for p in preds_lb[:k]})`, where `preds_lb` is `preds_raw` with the self-`question_id` removed and capped at `top_k`.
- **FR-6.2** The script SHALL accumulate three counter dicts (`qid_counters`, `proc_counters`, `proc_lb_counters`) keyed by `topK` for each mode.
- **FR-6.3** The tqdm postfix SHALL show `qid={n}/{total} proc={n}/{total}` at the first K in `match_at_k`, mirroring v2:411-413.

### FR-7 — Output: JSON

- **FR-7.1** The script SHALL write one JSON per mode to `<output_dir>/eval_<mode>.json` with shape:
  ```json
  {
    "model": {"embedding_model": "...", "embedding_dim": 768, "mode": "...",
              "collection": "...", "top_k": 5, "match_at_k": [1,3,5],
              "num_queries": N},
    "mode": "...",
    "results": [
      {
        "question_id": "...", "question": "...",
        "ground_truth_qid": "...", "ground_truth_proc": "...",
        "ground_truth_proc_full": "...",
        "predictions": [ {<normalized_pred>}, ... ],
        "matches": {
          "qid":     {"top1": bool, "top3": bool, "top5": bool},
          "proc":    {"top1": bool, "top3": bool, "top5": bool},
          "proc_lb": {"top1": bool, "top3": bool, "top5": bool}
        }
      }, ...
    ]
  }
  ```
- **FR-7.2** JSON SHALL be written with `indent=2`. Existing files SHALL be overwritten.
- **FR-7.3** `output_dir` SHALL be created if missing (`mkdir(parents=True, exist_ok=True)`).

### FR-8 — Output: CSV

- **FR-8.1** The script SHALL write one CSV per mode to `<output_dir>/eval_<mode>.csv`.
- **FR-8.2** CSV header (single row, in this order):
  ```
  question_id, question, ground_truth_proc,
  pred_1_source, pred_1_qid, pred_1_proc_or_doc,
  pred_1_section_or_logic_chunk, score_1,
  ... ×top_k ...,
  qid_top1_match, qid_top3_match, qid_top5_match,
  proc_top1_match, proc_top3_match, proc_top5_match,
  proc_lb_top1_match, proc_lb_top3_match, proc_lb_top5_match
  ```
- **FR-8.3** For each prediction `p`:
  - `pred_i_proc_or_doc = p.get("proc_id", "") if p["source"] == "pub242" else p.get("doc_id", "")`.
  - `pred_i_section_or_logic_chunk = p.get("section_number", "") if p["source"] == "pub242" else p.get("u_logic_chunk_id", "")`.
  - `score_i = f"{p['score']:.4f}"`.
- **FR-8.4** Missing predictions (e.g. fewer than `top_k` returned) SHALL pad each missing slot with `["", "", "", "", ""]`.
- **FR-8.5** Match flag columns SHALL be Python `bool` (`True` / `False`); they MUST be present even when 0 (e.g. `techbriefs_only` mode).

### FR-9 — Console summary

- **FR-9.1** After each mode the script SHALL print to stderr:
  ```
  [<mode>] N queries
    Top-1  qid: a/N  proc: b/N  proc_lb: c/N
    Top-3  ...
    Top-5  ...
  Output JSON: <path>
  Output CSV:  <path>
  ```
  with `a/N` formatted as raw count over total (no percentage; mirrors v2:457-460 minus the percentage to keep it terse).

### FR-10 — CLI behavior

- **FR-10.1** `python eval.py` (no args) SHALL run the build (if needed) and all three modes against the full query set with `top_k=5`.
- **FR-10.2** `--rebuild` SHALL force `_build_db` regardless of collection state.
- **FR-10.3** `-n N` SHALL truncate the queries list to its first `N` entries before the eval loop. SHALL NOT affect the collection build.
- **FR-10.4** `--top-k K` SHALL override `cfg.eval.top_k` for retrieval; `match_at_k` is unchanged (filtered to `[k for k in match_at_k if k <= top_k]` to avoid out-of-range columns).
- **FR-10.5** `--mode <name>` SHALL run a single mode. `--mode all` (default) SHALL run every entry of `cfg.eval.modes`.
- **FR-10.6** Exit code 0 on successful runs (including `-n 0` which produces empty result lists). Non-zero on missing config keys, missing input files, malformed JSON inputs, or invalid `--mode`.

---

## 4. Non-Functional Requirements

### NFR-1 — Performance

- **NFR-1.1** Build (one-time, ≈49k passages, bge-base) SHALL complete in under 10 minutes on a CPU-only laptop and under 60 seconds on a single mid-range GPU. Dominant cost is the model forward pass; Qdrant upsert is sub-second per 500-batch.
- **NFR-1.2** Per-query retrieval SHALL be `O(log N + K)` (Qdrant HNSW); with `~50k` points, end-to-end latency SHOULD be under 50 ms per mode on CPU after the model is warm.
- **NFR-1.3** Memory footprint SHALL be `O(corpus_size × dim)` during the build (≈150 MB for 49k × 768 floats) plus a single SentenceTransformer model resident.

### NFR-2 — Reliability

- **NFR-2.1** The classifier mapping `mode -> Filter` SHALL be deterministic; same input → same Qdrant query.
- **NFR-2.2** Re-running `python eval.py` SHALL produce the same results modulo file modification times — Qdrant local-disk indexes are deterministic for a fixed insertion order.
- **NFR-2.3** A single bad row in `pub242_hq.json` (e.g. malformed `process_source_metadata`) SHALL NOT abort the build. The row's enrichment fields fall back to `""` and a stderr WARNING is emitted.
- **NFR-2.4** The script SHALL NOT swallow exceptions silently. Any unexpected error SHALL be logged with traceback and propagate as a non-zero exit.

### NFR-3 — Maintainability

- **NFR-3.1** The script SHALL be self-contained — copies of the v2 helpers listed in §1.4 are inlined. No imports from `examples/eval2/predict_pub242_v2.py`. No `aisa.*` imports.
- **NFR-3.2** `_normalize_pred`, `_make_filter`, `_load_pub242_docs`, `_load_techbriefs_docs`, `_parse_pub242_metadata` SHALL be testable in isolation (pure functions over JSON-shaped data).
- **NFR-3.3** Mode names SHALL be string constants defined at module scope so callers can reference them and they appear in `--help`.
- **NFR-3.4** The CSV header order SHALL be deterministic and match the row construction order so the file is `csv.DictReader`-friendly downstream.

### NFR-4 — Compatibility

- **NFR-4.1** Python 3.11+ (uses `tomllib`); the standard `tomli` shim SHALL be imported as a fallback to keep the script runnable on 3.10 if ever needed.
- **NFR-4.2** No new entries SHALL be added to `reqs.txt`. All required dependencies are already present for `predict_pub242_v2.py`.
- **NFR-4.3** The script SHALL be runnable from any CWD; relative paths in TOML resolve against the config file's parent.

---

## 5. Acceptance Criteria

- **AC-1** `eval.py` exists at `/Users/igor/dev/llm/pavement-gpt/nvidia-pipeline/nvidia-sdq_custom/eval.py` and is executable via `python eval.py`.
- **AC-2** `eval.toml` exists at the repo root with `[corpus]`, `[queries]`, `[embedding]`, `[qdrant]`, `[eval]` tables containing the keys listed in FR-1.4 and FR-1.5.
- **AC-3** First run with `-n 20` produces under `./output/`: `qdrant_db/` (Qdrant local store) plus `eval_pub242_only.{json,csv}`, `eval_techbriefs_only.{json,csv}`, `eval_both.{json,csv}` — six output files.
- **AC-4** `jq '.results[].predictions[].source' output/eval_pub242_only.json | sort -u` returns only `"pub242"`.
- **AC-5** `jq '.results[].predictions[].source' output/eval_techbriefs_only.json | sort -u` returns only `"techbriefs"`.
- **AC-6** `jq '.results[].predictions[].source' output/eval_both.json | sort -u` returns a subset of `{"pub242", "techbriefs"}` (typically both).
- **AC-7** Re-running `python eval.py -n 20` (no `--rebuild`) prints `[ensure_db] DB ready: collection='answers_combined', points=49xxx, dim=768` and skips encoding (verifiable by absence of the build tqdm bar).
- **AC-8** `pub242_only` `qid_top1` count matches the v2 `[rag]` baseline at `examples/eval2/output_v2/pub242_rag_results.json` to within 0 (same vectors + cosine + same K).
- **AC-9** Each result row in any mode's JSON has `predictions` of length ≤ `top_k` and a `matches` dict with the three families and the three K values per family.
- **AC-10** Each CSV row has the same number of fields as the header (`awk -F, '{print NF}' <csv> | sort -u` yields exactly one value).
- **AC-11** `techbriefs_only` JSON has all `matches.qid.*`, `matches.proc.*`, `matches.proc_lb.*` set to `false` (no pub242 records in the pool).
- **AC-12** Running with an invalid `--mode foo` exits non-zero and prints a usage error.
- **AC-13** Running with `[corpus] pub242_json` pointing at a missing file exits non-zero and prints an ERROR naming the missing path. No collection is created.
- **AC-14** The script does not import from `aisa.*` and does not require `OPENAI_API_KEY` / `GOOGLE_API_KEY`. Verified by `grep -E "from aisa|OPENAI_API_KEY|GOOGLE_API_KEY" eval.py` returning nothing relevant.
- **AC-15** Running `python eval.py --rebuild -n 20` deletes and recreates the collection, then runs all three modes, leaving the same six output files updated.

---

## 6. Out-of-scope follow-ups

- **OQ-1** Add a `correlate` sub-command analogous to v2's `run_correlate` — produce a unified table comparing `pub242_only` vs `both` (does adding techbriefs degrade self-retrieval?).
- **OQ-2** Add a fourth mode `pub242_minus_self` that filters out the self-match payload before retrieval. Today the same effect is achieved post-hoc by `proc_lb`.
- **OQ-3** Add reranking via a small cross-encoder. Useful especially for `both` mode where the densely-populated techbriefs pool may bury the relevant pub242 hit.
- **OQ-4** Add a `[corpus] limit_techbriefs = N` knob for fast iteration during prompt or chunking changes. User explicitly opted out for v1; revisit if iteration time becomes painful.
- **OQ-5** Add hybrid search (BM25 + dense) — Qdrant supports it natively as of recent releases.
- **OQ-6** Add a chapter / section-level breakdown of metrics (e.g. proc-match rate by `chapter_number`) — would require pivoting the JSON downstream.
- **OQ-7** Asymmetric-model swap path. Today bge-base is symmetric, so `encode_query` is a no-op. Swapping in `Qwen3-Embedding-0.6B` would make the asymmetry meaningful — the `hasattr` fallback already routes correctly, but document encoding would need to switch to `encode_document` (which `_encode_documents_batched` already supports).

---

## 7. Decisions flagged

- **D-1 Single Qdrant collection, payload filter for modes.** Guarantees identical vectors across modes; one build, one model load, one disk path. Two collections would let us tune `dim` per source or use different embedders, but neither is needed.
- **D-2 Skip FAISS / Ollama / Unsloth backends.** The new question is "what does the candidate pool look like?", not "what backend?". The v2 script already exercises those backends.
- **D-3 `match_at_k = [1, 3, 5]`.** User requested top1/top3/top5 explicitly. Cuts CSV columns vs v2's `[1..5]` and matches natural reporting buckets.
- **D-4 `ast.literal_eval` first, `json.loads` second for `process_source_metadata`.** Current data is Python-literal repr (single quotes); `json.loads` would reject it. `ast.literal_eval` is safe (literals only). The JSON fallback is a hedge for future inputs.
- **D-5 Helpers copied from v2, not imported.** Keeps `eval.py` standalone and matches `filter-questions-citation-eval.py`'s convention. The duplication is small (≈80 lines) and makes the script grep-friendly.
- **D-6 Preserve v2's `proc_lb` metric.** Even in `pub242_only` self-retrieval where rank-1 is almost always the self-match, the low-boundary view ("can we find the right process via *other* answers?") is still informative; cost is one extra `[:top_k]` slice.
- **D-7 No reranking, no hybrid search, no asymmetric prompt swap in v1.** Out of scope; can be added once we know the baseline numbers.
- **D-8 Encode entire combined corpus in a single `embed_st_documents_batch` call.** Single tqdm bar, single model warm-up, no per-source bookkeeping. Source identity lives in the doc dict, which becomes the Qdrant payload.
- **D-9 Stable point IDs via `uuid5("source:question_id")`.** Namespacing by source means a hypothetical cross-corpus `question_id` collision (none observed empirically) cannot overwrite either point.
