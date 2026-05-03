"""
eval.py — pub242-vs-techbriefs retrieval evaluation.

Embeds two answer corpora (pub242 hq + techbriefs c-eval) into one Qdrant
collection, distinguished by a `source` payload field. Queries with pub242 hq
questions under three pool-restriction modes:

  - pub242_only      : only pub242 answers (mirrors v2 [rag] baseline)
  - techbriefs_only  : only techbriefs answers
  - both             : unified pool (single ranking)

Modes are Qdrant Filter clauses on `source`; vectors are bit-identical across
modes.

See plans/plan-eval-pub242-vs-techbriefs.md and plans/srs-eval-pub242-vs-techbriefs.md.
"""

import argparse
import ast
import csv
import json
import random
import re
import sys
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


# ---------------------------------------------------------------------------
# Mode constants
# ---------------------------------------------------------------------------
MODE_PUB242_ONLY     = "pub242_only"
MODE_TECHBRIEFS_ONLY = "techbriefs_only"
MODE_BOTH            = "both"
ALL_MODES = (MODE_PUB242_ONLY, MODE_TECHBRIEFS_ONLY, MODE_BOTH)

SELECTION_SEQUENTIAL = "sequential"
SELECTION_RANDOM     = "random"
ALL_SELECTIONS = (SELECTION_SEQUENTIAL, SELECTION_RANDOM)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
_PATH_KEYS = {
    "corpus":   ("pub242_json", "techbriefs_json"),
    "queries":  ("input",),
    "qdrant":   ("db_path",),
    "eval":     ("output_dir",),
}


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    base = config_path.parent

    for section, keys in _PATH_KEYS.items():
        if section not in cfg:
            continue
        for key in keys:
            if key in cfg[section]:
                p = Path(cfg[section][key])
                if not p.is_absolute():
                    cfg[section][key] = str((base / p).resolve())

    # Required keys
    required = [
        ("corpus",    "pub242_json"),
        ("corpus",    "techbriefs_json"),
        ("queries",   "input"),
        ("embedding", "model"),
        ("embedding", "dim"),
        ("qdrant",    "db_path"),
        ("qdrant",    "collection"),
    ]
    missing = [f"[{s}] {k}" for s, k in required
               if s not in cfg or k not in cfg[s]]
    if missing:
        print(f"Error: missing config keys: {', '.join(missing)}",
              file=sys.stderr)
        sys.exit(1)

    cfg.setdefault("eval", {})
    cfg["eval"].setdefault("top_k", 5)
    cfg["eval"].setdefault("match_at_k", [1, 3, 5])
    cfg["eval"].setdefault("output_dir",
                           str((base / "output").resolve()))
    cfg["eval"].setdefault("modes", list(ALL_MODES))
    cfg["embedding"].setdefault("batch_size", 64)

    cfg.setdefault("queries", {})
    cfg["queries"].setdefault("num_questions", 0)
    cfg["queries"].setdefault("selection", SELECTION_SEQUENTIAL)
    cfg["queries"].setdefault("seed", 42)

    bad_modes = [m for m in cfg["eval"]["modes"] if m not in ALL_MODES]
    if bad_modes:
        print(f"Error: invalid modes in config: {bad_modes}. "
              f"Valid: {list(ALL_MODES)}", file=sys.stderr)
        sys.exit(1)

    if cfg["queries"]["selection"] not in ALL_SELECTIONS:
        print(f"Error: invalid [queries] selection="
              f"{cfg['queries']['selection']!r}. "
              f"Valid: {list(ALL_SELECTIONS)}", file=sys.stderr)
        sys.exit(1)

    return cfg


# ---------------------------------------------------------------------------
# SentenceTransformers helpers (copied verbatim from
# examples/eval2/predict_pub242_v2.py — keeps eval.py standalone)
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


def extract_proc_id(process_id: str) -> str:
    """Convert 'publication_242-..._section_6.1-proc-0' → '6.1-0'."""
    m = re.search(r"section_([\d.]+)-proc-(\d+)", process_id)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m = re.search(r"chapter_(\d+)_intro-proc-(\d+)", process_id)
    if m:
        return f"{m.group(1)}.0-{m.group(2)}"
    return process_id


def _answer_snippet(text, n: int = 200) -> str:
    if text is None:
        return ""
    s = str(text)
    return s[:n]


# ---------------------------------------------------------------------------
# Doc loading
# ---------------------------------------------------------------------------
def _parse_pub242_metadata(s) -> dict:
    """Parse `process_source_metadata` (Python-literal string) into a dict.

    Robust: tries ast.literal_eval first, then json.loads. On failure logs a
    stderr WARNING and returns {} so a single bad row doesn't abort the build.
    """
    if not s or not isinstance(s, str):
        return {}
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, dict) else {}
    except (ValueError, SyntaxError):
        pass
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except (json.JSONDecodeError, TypeError):
        print(f"[pub242] metadata parse failed: {s[:80]}...", file=sys.stderr)
        return {}


def _load_pub242_docs(path: Path) -> list[dict]:
    if not path.exists():
        print(f"Error: pub242 corpus not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        print(f"Error: pub242 corpus must be a JSON list: {path}",
              file=sys.stderr)
        sys.exit(1)

    docs = []
    for item in data:
        answer = (item.get("answer") or "").strip()
        if not answer:
            continue
        process_id = item.get("process_id", "")
        md = _parse_pub242_metadata(item.get("process_source_metadata", ""))
        docs.append({
            "source":          "pub242",
            "question_id":     item["question_id"],
            "answer":          answer,
            "process_id":      process_id,
            "proc_id":         extract_proc_id(process_id),
            "source_chunk_id": item.get("source_chunk_id", ""),
            "chapter_number": str(md.get("chapter_number", "")),
            "chapter_title":  md.get("chapter_title", ""),
            "section_number": str(md.get("section_number", "")),
            "section_title":  md.get("section_title", ""),
            "source_id":      md.get("source_id", ""),
            "source_title":   md.get("source_title_short")
                              or md.get("source_title", ""),
        })
    print(f"[load] pub242: kept {len(docs)} / total {len(data)}",
          file=sys.stderr)
    return docs


def _load_techbriefs_docs(path: Path) -> list[dict]:
    if not path.exists():
        print(f"Error: techbriefs corpus not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        print(f"Error: techbriefs corpus must be a JSON list: {path}",
              file=sys.stderr)
        sys.exit(1)

    docs = []
    for item in data:
        answer = (item.get("answer") or "").strip()
        if not answer:
            continue
        art = item.get("artifact") or {}
        docs.append({
            "source":               "techbriefs",
            "question_id":          item["question_id"],
            "answer":               answer,
            "doc_id":               item.get("doc_id", ""),
            "u_ctx_id":             item.get("u_ctx_id", ""),
            "u_logic_chunk_id":     item.get("u_logic_chunk_id", ""),
            "source_u_chunk_ids":   item.get("source_u_chunk_ids", []) or [],
            "artifact_id":          item.get("artifact_id", ""),
            "u_artifact_id":        item.get("u_artifact_id", ""),
            "artifact_description": art.get("description", ""),
        })
    print(f"[load] techbriefs: kept {len(docs)} / total {len(data)}",
          file=sys.stderr)
    return docs


# ---------------------------------------------------------------------------
# Database build / load
# ---------------------------------------------------------------------------
def _upsert_batched(client, collection: str, points: list, batch: int = 500):
    for i in range(0, len(points), batch):
        client.upsert(collection_name=collection, points=points[i:i + batch])


def _build_db(client, cfg: dict):
    from qdrant_client.models import Distance, VectorParams, PointStruct

    collection  = cfg["qdrant"]["collection"]
    dim         = int(cfg["embedding"]["dim"])
    embed_model = cfg["embedding"]["model"]
    batch_size  = int(cfg["embedding"]["batch_size"])

    pub_docs = _load_pub242_docs(Path(cfg["corpus"]["pub242_json"]))
    tbf_docs = _load_techbriefs_docs(Path(cfg["corpus"]["techbriefs_json"]))
    docs = pub_docs + tbf_docs
    print(f"[build] {len(pub_docs)} pub242 + {len(tbf_docs)} techbriefs = "
          f"{len(docs)} total", file=sys.stderr)

    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    print(f"[build] encoding {len(docs)} answers with "
          f"model={embed_model}, batch_size={batch_size}", file=sys.stderr)
    texts = [d["answer"] for d in docs]
    emb_matrix = embed_st_documents_batch(texts, embed_model,
                                           batch_size=batch_size)

    points = []
    for doc, emb in zip(docs, emb_matrix):
        pid = str(uuid.uuid5(
            uuid.NAMESPACE_DNS,
            f'{doc["source"]}:{doc["question_id"]}'
        ))
        points.append(PointStruct(
            id=pid,
            vector=emb.tolist(),
            payload=doc,
        ))

    _upsert_batched(client, collection, points)
    print(f"[build] upserted {len(points)} points to '{collection}'",
          file=sys.stderr)


def ensure_db(cfg: dict, force: bool = False):
    from qdrant_client import QdrantClient

    db_path    = cfg["qdrant"]["db_path"]
    collection = cfg["qdrant"]["collection"]

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
        _build_db(client, cfg)
    else:
        info = client.get_collection(collection)
        print(f"[ensure_db] DB ready: collection='{collection}', "
              f"points={info.points_count}, "
              f"dim={info.config.params.vectors.size}", file=sys.stderr)
    return client


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def _make_filter(mode: str):
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    if mode == MODE_PUB242_ONLY:
        return Filter(must=[FieldCondition(
            key="source", match=MatchValue(value="pub242"))])
    if mode == MODE_TECHBRIEFS_ONLY:
        return Filter(must=[FieldCondition(
            key="source", match=MatchValue(value="techbriefs"))])
    if mode == MODE_BOTH:
        return None
    raise ValueError(f"unknown mode: {mode}")


def _normalize_pred(rank: int, hit) -> dict:
    p = hit.payload or {}
    out = {
        "rank":           rank,
        "score":          float(hit.score),
        "source":         p.get("source", ""),
        "question_id":    p.get("question_id", ""),
        "answer_snippet": _answer_snippet(p.get("answer", "")),
    }
    if p.get("source") == "pub242":
        out.update({
            "proc_id":        p.get("proc_id", ""),
            "process_id":     p.get("process_id", ""),
            "chapter_number": p.get("chapter_number", ""),
            "chapter_title":  p.get("chapter_title", ""),
            "section_number": p.get("section_number", ""),
            "section_title":  p.get("section_title", ""),
        })
    elif p.get("source") == "techbriefs":
        out.update({
            "doc_id":             p.get("doc_id", ""),
            "u_ctx_id":           p.get("u_ctx_id", ""),
            "u_logic_chunk_id":   p.get("u_logic_chunk_id", ""),
            "source_u_chunk_ids": p.get("source_u_chunk_ids", []),
            "artifact_id":        p.get("artifact_id", ""),
            "u_artifact_id":      p.get("u_artifact_id", ""),
        })
    return out


def _retrieve(client, collection: str, qvec: np.ndarray, k: int,
              mode: str) -> list[dict]:
    hits = client.query_points(
        collection_name=collection,
        query=qvec.tolist(),
        query_filter=_make_filter(mode),
        limit=k,
        with_payload=True,
    ).points
    return [_normalize_pred(rank, h) for rank, h in enumerate(hits, 1)]


# ---------------------------------------------------------------------------
# Per-mode eval loop
# ---------------------------------------------------------------------------
def _eval_one_mode(cfg: dict, client, queries: list[dict], mode: str,
                   query_selection_info: Optional[dict] = None):
    top_k       = int(cfg["eval"]["top_k"])
    match_at_k  = [int(k) for k in cfg["eval"]["match_at_k"] if int(k) <= top_k]
    output_dir  = Path(cfg["eval"]["output_dir"])
    collection  = cfg["qdrant"]["collection"]
    embed_model = cfg["embedding"]["model"]

    output_dir.mkdir(parents=True, exist_ok=True)

    qid_counters     = {f"top{k}": 0 for k in match_at_k}
    proc_counters    = {f"top{k}": 0 for k in match_at_k}
    proc_lb_counters = {f"top{k}": 0 for k in match_at_k}

    results: list[dict] = []
    pbar = tqdm(queries, desc=f"[eval:{mode}]", unit="q",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}] {postfix}")
    for item in pbar:
        qid       = item["question_id"]
        question  = item["question"]
        gt_full   = item.get("process_id", "")
        gt_proc   = extract_proc_id(gt_full)

        qvec = embed_st_query(question, embed_model)
        preds_raw = _retrieve(client, collection, qvec, top_k + 1, mode)
        preds = preds_raw[:top_k]

        qid_preds  = [p["question_id"] for p in preds]
        proc_preds = [p.get("proc_id", "") for p in preds]
        qid_match  = {f"top{k}": qid in qid_preds[:k] for k in match_at_k}
        proc_match = {f"top{k}": bool(gt_proc) and gt_proc in proc_preds[:k]
                      for k in match_at_k}

        preds_lb = [p for p in preds_raw if p["question_id"] != qid][:top_k]
        proc_lb_preds = [p.get("proc_id", "") for p in preds_lb]
        proc_lb_match = {f"top{k}": bool(gt_proc)
                         and gt_proc in proc_lb_preds[:k]
                         for k in match_at_k}

        for k in match_at_k:
            key = f"top{k}"
            qid_counters[key]     += int(qid_match[key])
            proc_counters[key]    += int(proc_match[key])
            proc_lb_counters[key] += int(proc_lb_match[key])

        results.append({
            "question_id":            qid,
            "question":               question,
            "ground_truth_qid":       qid,
            "ground_truth_proc":      gt_proc,
            "ground_truth_proc_full": gt_full,
            "predictions":            preds,
            "matches": {
                "qid":     qid_match,
                "proc":    proc_match,
                "proc_lb": proc_lb_match,
            },
        })

        n = len(results)
        first_k = f"top{match_at_k[0]}" if match_at_k else "top1"
        pbar.set_postfix_str(
            f"qid={qid_counters.get(first_k, 0)}/{n} "
            f"proc={proc_counters.get(first_k, 0)}/{n}"
        )

    model_info = {
        "embedding_model": embed_model,
        "embedding_dim":   int(cfg["embedding"]["dim"]),
        "mode":            mode,
        "collection":      collection,
        "top_k":           top_k,
        "match_at_k":      match_at_k,
        "num_queries":     len(results),
    }
    if query_selection_info:
        model_info["query_selection"] = query_selection_info
    json_path = output_dir / f"eval_{mode}.json"
    csv_path  = output_dir / f"eval_{mode}.csv"
    _write_json(json_path, model_info, mode, results)
    _write_csv(csv_path, results, top_k, match_at_k)

    total = len(results)
    print(f"\n[{mode}] {total} queries", file=sys.stderr)
    for k in match_at_k:
        key = f"top{k}"
        print(f"  Top-{k}  qid: {qid_counters[key]}/{total}  "
              f"proc: {proc_counters[key]}/{total}  "
              f"proc_lb: {proc_lb_counters[key]}/{total}", file=sys.stderr)
    print(f"Output JSON: {json_path}", file=sys.stderr)
    print(f"Output CSV:  {csv_path}",  file=sys.stderr)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def _write_json(path: Path, model_info: dict, mode: str,
                results: list[dict]):
    with open(path, "w") as f:
        json.dump({"model": model_info, "mode": mode, "results": results},
                  f, indent=2)


def _write_csv(path: Path, results: list[dict], top_k: int,
               match_at_k: list[int]):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["question_id", "question", "ground_truth_proc"]
        for i in range(1, top_k + 1):
            header.extend([
                f"pred_{i}_source",
                f"pred_{i}_qid",
                f"pred_{i}_proc_or_doc",
                f"pred_{i}_section_or_logic_chunk",
                f"score_{i}",
            ])
        for k in match_at_k:
            header.append(f"qid_top{k}_match")
        for k in match_at_k:
            header.append(f"proc_top{k}_match")
        for k in match_at_k:
            header.append(f"proc_lb_top{k}_match")
        writer.writerow(header)

        for r in results:
            row = [r["question_id"], r["question"], r["ground_truth_proc"]]
            preds = r["predictions"]
            for p in preds:
                src = p.get("source", "")
                if src == "pub242":
                    proc_or_doc = p.get("proc_id", "")
                    section_or_logic = p.get("section_number", "")
                else:
                    proc_or_doc = p.get("doc_id", "")
                    section_or_logic = p.get("u_logic_chunk_id", "")
                row.extend([
                    src,
                    p.get("question_id", ""),
                    proc_or_doc,
                    section_or_logic,
                    f"{p['score']:.4f}",
                ])
            for _ in range(top_k - len(preds)):
                row.extend(["", "", "", "", ""])
            for k in match_at_k:
                row.append(r["matches"]["qid"][f"top{k}"])
            for k in match_at_k:
                row.append(r["matches"]["proc"][f"top{k}"])
            for k in match_at_k:
                row.append(r["matches"]["proc_lb"][f"top{k}"])
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------
def _load_queries(path: Path, limit: Optional[int],
                  selection: str, seed: int) -> list[dict]:
    """Load the query JSON, then sample / truncate to `limit` entries.

    `limit` of None or <= 0 means "use all loaded queries".
    `selection` is one of `sequential` (first N) or `random` (uniform sample,
    seeded for reproducibility).
    """
    if not path.exists():
        print(f"Error: queries file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        queries = json.load(f)
    if not isinstance(queries, list):
        print(f"Error: queries file must be a JSON list: {path}",
              file=sys.stderr)
        sys.exit(1)

    total = len(queries)
    if limit is None or limit <= 0 or limit >= total:
        print(f"[queries] using all {total} loaded queries", file=sys.stderr)
        return queries

    if selection == SELECTION_SEQUENTIAL:
        chosen = queries[:limit]
        print(f"[queries] sequential: first {limit}/{total}", file=sys.stderr)
    elif selection == SELECTION_RANDOM:
        chosen = random.Random(seed).sample(queries, limit)
        print(f"[queries] random: sampled {limit}/{total} (seed={seed})",
              file=sys.stderr)
    else:
        # Already validated in load_config; defensive fallback.
        print(f"Error: unknown selection {selection!r}", file=sys.stderr)
        sys.exit(1)
    return chosen


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    default_cfg = Path(__file__).resolve().parent / "eval.toml"

    parser = argparse.ArgumentParser(
        description="pub242-vs-techbriefs retrieval evaluation"
    )
    parser.add_argument("--cfg", type=Path, default=default_cfg,
                        help=f"Path to TOML config (default: {default_cfg})")
    parser.add_argument("--rebuild", action="store_true",
                        help="Drop and rebuild the Qdrant collection")
    parser.add_argument("--mode", default="all",
                        choices=list(ALL_MODES) + ["all"],
                        help="Eval mode (default: all -> every entry of "
                             "[eval] modes)")
    parser.add_argument("-n", type=int, default=None,
                        help="Limit eval to N queries (overrides "
                             "[queries] num_questions)")
    parser.add_argument("--selection", default=None,
                        choices=list(ALL_SELECTIONS),
                        help="Override [queries] selection")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override [queries] seed (only used by "
                             "selection=random)")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Override [eval] top_k")
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    if args.top_k is not None:
        cfg["eval"]["top_k"] = int(args.top_k)

    limit = args.n if args.n is not None \
        else int(cfg["queries"].get("num_questions", 0))
    selection = args.selection or cfg["queries"]["selection"]
    seed = args.seed if args.seed is not None \
        else int(cfg["queries"].get("seed", 42))

    client = ensure_db(cfg, force=args.rebuild)

    queries = _load_queries(
        Path(cfg["queries"]["input"]),
        limit=limit, selection=selection, seed=seed,
    )

    if args.mode == "all":
        modes = list(cfg["eval"]["modes"])
    else:
        modes = [args.mode]

    query_selection_info = {
        "selection":     selection,
        "limit":         limit if limit and limit > 0 else None,
        "seed":          seed if selection == SELECTION_RANDOM else None,
        "num_selected":  len(queries),
    }

    for mode in modes:
        _eval_one_mode(cfg, client, queries, mode,
                       query_selection_info=query_selection_info)


if __name__ == "__main__":
    main()
