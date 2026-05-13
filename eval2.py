"""
eval2.py — base vs fine-tuned embedding eval on the techbriefs corpus.

Corpus: all answers from the techbriefs eval JSON, keyed by question_id.
Query:  the question text.
Compared methods (Qdrant only):
  - rag    : baseline model (default: BAAI/bge-base-en-v1.5)
  - rag-ft : fine-tuned model (default: ./examples/eval2/bge-base-pavements-matryoshka)

Match metrics per question (top-1, top-3, top-5):
  - qid         : did the predicted point share question_id?
                  (sanity check; trivial without self-exclusion)
  - logic_chunk : did the predicted point share u_logic_chunk_id, AFTER
                  excluding the self-match (predicted question_id == GT)?
  - ctx         : did the predicted point share u_ctx_id, AFTER excluding
                  the self-match?

Sibling of predict_pub242_v2.py — slimmed to the two methods this comparison
needs and re-targeted at techbriefs metadata. Path resolution differs: paths
in the TOML are CWD-relative (run from project root), not config-relative.
"""

import argparse
import csv
import json
import sys
import uuid
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


# ---------------------------------------------------------------------------
# Config loading (CWD-relative)
# ---------------------------------------------------------------------------
_GENERAL_PATH_KEYS = ("input", "db_path", "output_dir")
_METHOD_CONDITIONAL_PATH_KEYS = ("embedding_model",)
_METHOD_SECTIONS = ("rag", "rag-ft")


def load_config(config_path: Path) -> dict:
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    general = cfg.setdefault("general", {})
    for key in _GENERAL_PATH_KEYS:
        if key in general:
            p = Path(general[key])
            if not p.is_absolute():
                general[key] = str(p.resolve())

    for section in _METHOD_SECTIONS:
        if section not in cfg:
            continue
        for key in _METHOD_CONDITIONAL_PATH_KEYS:
            if key in cfg[section]:
                p = Path(cfg[section][key])
                if not p.is_absolute():
                    resolved = p.resolve()
                    if resolved.exists():
                        cfg[section][key] = str(resolved)

    return cfg


def _derived_outputs(cfg: dict, section: str) -> tuple[Path, Path]:
    out_dir = Path(cfg["general"]["output_dir"])
    name = cfg["general"].get("name", "techbriefs")
    suffix = section.replace("-", "_")
    return (out_dir / f"{name}_{suffix}_results.json",
            out_dir / f"{name}_{suffix}_results.csv")


# ---------------------------------------------------------------------------
# Embedding helpers (sentence-transformers; same pattern as predict_pub242_v2)
# ---------------------------------------------------------------------------
_st_cache: dict = {}


def _get_st_model(model_path: str):
    if model_path not in _st_cache:
        from sentence_transformers import SentenceTransformer
        _st_cache[model_path] = SentenceTransformer(model_path)
    return _st_cache[model_path]


def embed_st_query(text: str, model_path: str) -> np.ndarray:
    model = _get_st_model(model_path)
    if hasattr(model, "encode_query"):
        return model.encode_query(text, convert_to_numpy=True).astype(np.float32)
    return model.encode(text, convert_to_numpy=True).astype(np.float32)


def _encode_documents_batched(model, texts: list[str], batch_size: int) -> np.ndarray:
    """Prefer encode_document when available (asymmetric models), else encode."""
    if hasattr(model, "encode_document"):
        arr = model.encode_document(
            texts,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=True,
        )
    else:
        arr = model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=True,
        )
    return np.asarray(arr, dtype=np.float32)


def embed_st_documents_batch(texts: list[str], model_path: str,
                              batch_size: int = 64) -> np.ndarray:
    model = _get_st_model(model_path)
    return _encode_documents_batched(model, texts, batch_size)


def embed_ollama(text: str, model: str, base_url: str) -> np.ndarray:
    import ollama
    client = ollama.Client(host=base_url)
    resp = client.embed(model=model, input=text)
    return np.array(resp["embeddings"][0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
_REQUIRED_DOC_FIELDS = ("question_id", "doc_id", "u_ctx_id", "u_logic_chunk_id")


def _load_answer_docs(questions_json_path: Path) -> list[dict]:
    """Read techbriefs eval JSON; return one record per non-empty answer.

    Each record contains the routing metadata that gets stored in the Qdrant
    payload so retrieved hits can be resolved back to their source.
    """
    if not questions_json_path.exists():
        print(f"Error: questions JSON not found: {questions_json_path}",
              file=sys.stderr)
        sys.exit(1)

    with open(questions_json_path) as f:
        data = json.load(f)

    docs = []
    skipped = 0
    for item in data:
        answer = (item.get("answer") or "").strip()
        if not answer:
            skipped += 1
            continue
        docs.append({
            "question_id":      item["question_id"],
            "doc_id":           item.get("doc_id", ""),
            "u_ctx_id":         item.get("u_ctx_id", ""),
            "u_logic_chunk_id": item.get("u_logic_chunk_id", ""),
            "answer":           answer,
        })
    if skipped:
        print(f"[load] dropped {skipped} records with empty answer",
              file=sys.stderr)
    return docs


def _load_questions(input_path: Path, limit: int | None) -> list[dict]:
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    with open(input_path) as f:
        questions = json.load(f)
    questions = [q for q in questions if (q.get("answer") or "").strip()]
    if limit is not None:
        questions = questions[:limit]
    return questions


def _answer_snippet(text, n: int = 200) -> str:
    if text is None:
        return ""
    return str(text)[:n]


def _make_model_info(cfg: dict, method: str, embed_model: str,
                     extras: dict) -> dict:
    info = {
        "name": cfg["general"].get("name", "techbriefs"),
        "method": method,
        "version": "v2",
        "eval_mode": "self_retrieval",
        "doc_source": "answers",
        "embedding_model": embed_model,
    }
    info.update(extras)
    return info


# ---------------------------------------------------------------------------
# Shared eval loop — five match families:
#   qid             — predicted question_id == GT, no exclusion
#   logic_chunk     — predicted u_logic_chunk_id == GT, no exclusion
#   lb_logic_chunk  — same, but after dropping the self-match
#   ctx             — predicted u_ctx_id == GT, no exclusion
#   lb_ctx          — same, but after dropping the self-match
# ---------------------------------------------------------------------------
_MATCH_FAMILIES = ("qid", "logic_chunk", "lb_logic_chunk", "ctx", "lb_ctx")


def _run_eval(cfg: dict, section: str, model_info: dict, retrieve_fn,
              top_k: int, output_json_path: Path, output_csv_path: Path,
              pbar_desc: str, limit: int | None = None):
    """Loop over questions, retrieve top_k+1, score, and write JSON+CSV.

    `retrieve_fn(question, k) -> [pred, ...]` where each pred has keys:
        rank, question_id, doc_id, u_ctx_id, u_logic_chunk_id, score,
        answer_snippet
    """
    match_at_k = cfg[section].get("match_at_k", [1, 3, 5])
    input_path = Path(cfg["general"]["input"])
    questions = _load_questions(input_path, limit)

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    results_all = []
    counters = {fam: {f"top{k}": 0 for k in match_at_k}
                for fam in _MATCH_FAMILIES}

    pbar = tqdm(questions, desc=pbar_desc, unit="q",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}] {postfix}")
    for item in pbar:
        question_id = item["question_id"]
        question = item["question"]
        gt = {
            "question_id":      question_id,
            "doc_id":           item.get("doc_id", ""),
            "u_ctx_id":         item.get("u_ctx_id", ""),
            "u_logic_chunk_id": item.get("u_logic_chunk_id", ""),
        }

        # Fetch one extra so self-exclusion still leaves top_k.
        preds_raw = retrieve_fn(question, top_k + 1)
        preds = preds_raw[:top_k]
        preds_excl = [p for p in preds_raw
                      if p["question_id"] != question_id][:top_k]

        qid_preds    = [p["question_id"]      for p in preds]
        lc_preds     = [p["u_logic_chunk_id"] for p in preds]
        ctx_preds    = [p["u_ctx_id"]         for p in preds]
        lb_lc_preds  = [p["u_logic_chunk_id"] for p in preds_excl]
        lb_ctx_preds = [p["u_ctx_id"]         for p in preds_excl]

        matches = {
            "qid":            {f"top{k}": question_id              in qid_preds[:k]    for k in match_at_k},
            "logic_chunk":    {f"top{k}": gt["u_logic_chunk_id"]   in lc_preds[:k]     for k in match_at_k},
            "lb_logic_chunk": {f"top{k}": gt["u_logic_chunk_id"]   in lb_lc_preds[:k]  for k in match_at_k},
            "ctx":            {f"top{k}": gt["u_ctx_id"]           in ctx_preds[:k]    for k in match_at_k},
            "lb_ctx":         {f"top{k}": gt["u_ctx_id"]           in lb_ctx_preds[:k] for k in match_at_k},
        }

        for fam in _MATCH_FAMILIES:
            for k in match_at_k:
                key = f"top{k}"
                counters[fam][key] += int(matches[fam][key])

        results_all.append({
            "question_id": question_id,
            "question": question,
            "ground_truth": gt,
            "predictions": preds,
            "matches": matches,
        })

        first_k = f"top{match_at_k[0]}"
        n = len(results_all)
        pbar.set_postfix_str(
            f"qid={counters['qid'][first_k]}/{n} "
            f"lc={counters['logic_chunk'][first_k]}/{n} "
            f"lb_lc={counters['lb_logic_chunk'][first_k]}/{n}"
        )

    total = len(questions)

    with open(output_json_path, "w") as f:
        json.dump({"model": model_info, "results": results_all}, f, indent=2)

    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["question_id", "question",
                  "gt_doc_id", "gt_u_ctx_id", "gt_u_logic_chunk_id"]
        for i in range(1, top_k + 1):
            header.extend([f"pred_{i}_qid", f"pred_{i}_doc_id",
                           f"pred_{i}_u_ctx_id", f"pred_{i}_u_logic_chunk_id",
                           f"score_{i}"])
        for fam in _MATCH_FAMILIES:
            for k in match_at_k:
                header.append(f"{fam}_top{k}_match")
        writer.writerow(header)

        for r in results_all:
            row = [r["question_id"], r["question"],
                   r["ground_truth"]["doc_id"],
                   r["ground_truth"]["u_ctx_id"],
                   r["ground_truth"]["u_logic_chunk_id"]]
            for p in r["predictions"]:
                row.extend([p["question_id"], p["doc_id"],
                            p["u_ctx_id"], p["u_logic_chunk_id"],
                            f"{p['score']:.4f}"])
            for _ in range(top_k - len(r["predictions"])):
                row.extend(["", "", "", "", ""])
            for fam in _MATCH_FAMILIES:
                for k in match_at_k:
                    row.append(r["matches"][fam][f"top{k}"])
            writer.writerow(row)

    print(f"\nDone. {total} questions processed ({pbar_desc}).",
          file=sys.stderr)
    for k in match_at_k:
        key = f"top{k}"
        parts = [f"Top-{k} "]
        for fam in _MATCH_FAMILIES:
            cnt = counters[fam][key]
            parts.append(f"{fam}: {cnt}/{total} = {cnt / total:.4f}")
        print("  " + "   ".join(parts), file=sys.stderr)
    print(f"Output JSON: {output_json_path}", file=sys.stderr)
    print(f"Output CSV:  {output_csv_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# RAG (Qdrant): build / ensure / batch
# ---------------------------------------------------------------------------
def _embed_docs(section_cfg: dict, texts: list[str]) -> np.ndarray:
    backend = section_cfg.get("embedding_backend", "sentence-transformers")
    embed_model = section_cfg["embedding_model"]
    if backend == "sentence-transformers":
        batch_size = int(section_cfg.get("encode_batch_size", 64))
        return embed_st_documents_batch(texts, embed_model,
                                         batch_size=batch_size)
    if backend == "ollama":
        base_url = section_cfg.get("ollama_base_url",
                                    "http://localhost:11434")
        out = []
        for t in tqdm(texts, desc="Embedding answers (ollama)", unit="ans"):
            out.append(embed_ollama(t, embed_model, base_url))
        return np.asarray(out, dtype=np.float32)
    raise ValueError(f"unknown embedding_backend: {backend!r}")


def _embed_query(section_cfg: dict, text: str) -> np.ndarray:
    backend = section_cfg.get("embedding_backend", "sentence-transformers")
    embed_model = section_cfg["embedding_model"]
    if backend == "sentence-transformers":
        return embed_st_query(text, embed_model)
    if backend == "ollama":
        base_url = section_cfg.get("ollama_base_url",
                                    "http://localhost:11434")
        return embed_ollama(text, embed_model, base_url)
    raise ValueError(f"unknown embedding_backend: {backend!r}")


def _upsert_batched(client, collection: str, points: list, batch: int = 500):
    for i in range(0, len(points), batch):
        client.upsert(collection_name=collection, points=points[i:i + batch])


def _build_rag_db(client, cfg: dict, section: str):
    from qdrant_client.models import Distance, VectorParams, PointStruct

    sec_cfg = cfg[section]
    collection = sec_cfg["collection_name"]
    dim = sec_cfg["embedding_dim"]
    embed_model = sec_cfg["embedding_model"]
    backend = sec_cfg.get("embedding_backend", "sentence-transformers")
    questions_json = Path(cfg["general"]["input"])

    docs = _load_answer_docs(questions_json)

    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    print(f"Building [{section}] DB: {len(docs)} answers, "
          f"model={embed_model}, backend={backend}, dim={dim}",
          file=sys.stderr)

    texts = [d["answer"] for d in docs]
    emb_matrix = _embed_docs(sec_cfg, texts)

    points = []
    for doc, emb in zip(docs, emb_matrix):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc["question_id"]))
        points.append(PointStruct(
            id=point_id,
            vector=emb.tolist(),
            payload={
                "question_id":      doc["question_id"],
                "doc_id":           doc["doc_id"],
                "u_ctx_id":         doc["u_ctx_id"],
                "u_logic_chunk_id": doc["u_logic_chunk_id"],
                "answer":           doc["answer"],
            },
        ))

    _upsert_batched(client, collection, points)
    print(f"Upserted {len(points)} points to collection '{collection}'",
          file=sys.stderr)


def ensure_rag_db(cfg: dict, section: str, force: bool = False):
    from qdrant_client import QdrantClient

    sec_cfg = cfg[section]
    db_path = cfg["general"]["db_path"]
    collection = sec_cfg["collection_name"]

    Path(db_path).mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=db_path)

    needs_build = force
    if not needs_build:
        if not client.collection_exists(collection):
            needs_build = True
        else:
            info = client.get_collection(collection)
            if info.points_count == 0:
                needs_build = True

    if needs_build:
        _build_rag_db(client, cfg, section)
    else:
        info = client.get_collection(collection)
        print(f"[{section}] DB ready: collection='{collection}', "
              f"points={info.points_count}, "
              f"dim={info.config.params.vectors.size}", file=sys.stderr)
    return client


def run_batch_rag(cfg: dict, section: str,
                  topk_override: int | None = None,
                  limit: int | None = None):
    sec_cfg = cfg[section]
    top_k = topk_override if topk_override is not None else sec_cfg.get("top_k", 5)
    embed_model = sec_cfg["embedding_model"]
    backend = sec_cfg.get("embedding_backend", "sentence-transformers")

    client = ensure_rag_db(cfg, section)
    collection = sec_cfg["collection_name"]
    col_info = client.get_collection(collection)

    output_json_path, output_csv_path = _derived_outputs(cfg, section)

    model_info = _make_model_info(cfg, section, embed_model, {
        "embedding_dim": sec_cfg.get("embedding_dim"),
        "embedding_backend": backend,
        "collection_name": collection,
        "num_points": col_info.points_count,
        "num_docs": col_info.points_count,
    })

    def retrieve(question: str, k: int) -> list[dict]:
        vec = _embed_query(sec_cfg, question)
        hits = client.query_points(
            collection_name=collection,
            query=vec.tolist(),
            limit=k,
            with_payload=True,
        ).points
        preds = []
        for rank, h in enumerate(hits, 1):
            payload = h.payload or {}
            preds.append({
                "rank": rank,
                "question_id":      payload.get("question_id", ""),
                "doc_id":           payload.get("doc_id", ""),
                "u_ctx_id":         payload.get("u_ctx_id", ""),
                "u_logic_chunk_id": payload.get("u_logic_chunk_id", ""),
                "score": float(h.score),
                "answer_snippet": _answer_snippet(payload.get("answer", "")),
            })
        return preds

    _run_eval(cfg, section, model_info, retrieve, top_k,
              output_json_path, output_csv_path,
              pbar_desc=f"[{section}] search", limit=limit)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    default_config = Path(__file__).resolve().parent / "eval2.toml"

    parser = argparse.ArgumentParser(
        description="Self-retrieval eval of base vs fine-tuned embedding "
                    "models on the techbriefs corpus."
    )
    parser.add_argument("--config", type=Path, default=default_config,
                        help=f"Path to TOML config (default: {default_config})")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Number of predictions (overrides config)")
    parser.add_argument("-n", type=int, default=None,
                        help="Limit batch to first N questions")

    parser.add_argument("--setup-rag", action="store_true",
                        help="Build/rebuild base-model Qdrant answers DB")
    parser.add_argument("--batch-rag", action="store_true",
                        help="Run base-model self-retrieval eval")
    parser.add_argument("--setup-rag-ft", action="store_true",
                        help="Build/rebuild fine-tuned-model Qdrant answers DB")
    parser.add_argument("--batch-rag-ft", action="store_true",
                        help="Run fine-tuned-model self-retrieval eval")

    args = parser.parse_args()

    if not any([args.setup_rag, args.batch_rag,
                args.setup_rag_ft, args.batch_rag_ft]):
        parser.error("provide one of --setup-rag / --batch-rag / "
                     "--setup-rag-ft / --batch-rag-ft")

    if not args.config.exists():
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(args.config)

    if args.setup_rag:
        ensure_rag_db(cfg, "rag", force=True)
        return
    if args.batch_rag:
        run_batch_rag(cfg, "rag", topk_override=args.top_k, limit=args.n)
        return
    if args.setup_rag_ft:
        ensure_rag_db(cfg, "rag-ft", force=True)
        return
    if args.batch_rag_ft:
        run_batch_rag(cfg, "rag-ft", topk_override=args.top_k, limit=args.n)
        return


if __name__ == "__main__":
    main()
