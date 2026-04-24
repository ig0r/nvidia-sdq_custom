import argparse
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Set

from aisa.utils import files
from aisa.gen.embed import Embedder, EmbedConfig


def filter_and_convert(
    sdg_output_path: str, quality_threshold: float
) -> List[Dict[str, Any]]:
    with open(sdg_output_path, "r", encoding="utf-8") as f:
        sdg_records = json.load(f)

    training_docs = []
    question_counter = 0

    for record in sdg_records:
        file_name = record["file_name"]
        chunks = {
            chunk["chunk_id"]: chunk["text"] for chunk in record.get("chunks", [])
        }
        qa_pairs = record.get("qa_pairs", [])
        evaluations = record.get("qa_evaluations", [])

        for i, qa in enumerate(qa_pairs):
            # Check Quality Threshold
            score = 0.0
            if i < len(evaluations):
                eval_data = evaluations[i]
                overall = eval_data.get("overall", {})
                score = overall.get("score", 0.0)

            if score < quality_threshold:
                continue

            # Collect target segments
            segment_ids = qa.get("segment_ids", [])
            pos_docs = []
            for sid in segment_ids:
                if sid in chunks:
                    pos_docs.append(
                        {"id": f"{file_name}_chunk_{sid}", "text": chunks[sid]}
                    )
            if not pos_docs:
                continue

            q_id = f"q{question_counter:06d}"
            question_counter += 1

            training_docs.append(
                {
                    "question_id": q_id,
                    "question": qa.get("question", ""),
                    "corpus_id": file_name,
                    "pos_doc": pos_docs,
                    "neg_doc": [],
                }
            )

    return training_docs


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)


def mine_hard_negatives(
    data: List[Dict[str, Any]],
    embedder: Embedder,
    top_k: int = 5,
    hard_neg_margin: float = 0.95,
) -> List[Dict[str, Any]]:
    print("Extracting unique passages for embeddings cache...")
    # Build unique passages lookup
    passage_text_to_id = {}
    passage_id_to_text = {}

    for row in data:
        for pd in row.get("pos_doc", []):
            p_text = pd["text"]
            p_id = pd["id"]
            if p_text not in passage_text_to_id:
                passage_text_to_id[p_text] = p_id
            if p_id not in passage_id_to_text:
                passage_id_to_text[p_id] = p_text

    unique_texts = list(passage_text_to_id.keys())
    if not unique_texts:
        print("No valid passages found! Cannot mine hard negatives.")
        return data

    print(f"Embedding {len(unique_texts)} passages...")
    passage_embeddings = embedder.partition_inputs(unique_texts, chunk_size=32)

    print(f"Embedding {len(data)} queries...")
    queries_text = [row["question"] for row in data]
    query_embeddings = embedder.partition_inputs(queries_text, chunk_size=32)

    print("Computing similarity matrix...")
    sim_matrix = cosine_similarity(query_embeddings, passage_embeddings)

    for i, row in enumerate(data):
        target_ids = {pd["id"] for pd in row["pos_doc"]}
        sims = sim_matrix[i]

        # Argsort descending
        top_indices = np.argsort(sims)[::-1]

        hard_negs = []
        for idx in top_indices:
            p_text = unique_texts[idx]
            p_id = passage_text_to_id[p_text]
            score = sims[idx]

            if p_id in target_ids:
                continue

            # NVIDIA Hard neg margin check: we only take items below the threshold margin
            if score <= hard_neg_margin:
                hard_negs.append({"id": p_id, "text": p_text, "score": float(score)})

            if len(hard_negs) >= top_k:
                break

        row["neg_doc"] = hard_negs

    return data


def unroll_pos_docs(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    print("Unrolling multi-pos documents...")
    unrolled = []
    for record in data:
        pos_docs = record.get("pos_doc", [])
        if len(pos_docs) <= 1:
            unrolled.append(record)
        else:
            base_q_id = record["question_id"]
            for idx, pd in enumerate(pos_docs):
                unrolled.append(
                    {
                        "question_id": f"{base_q_id}_{idx}",
                        "question": record["question"],
                        "corpus_id": record["corpus_id"],
                        "pos_doc": [pd],
                        "neg_doc": record.get("neg_doc", []),
                    }
                )
    return unrolled


def save_splits(
    data: List[Dict[str, Any]],
    out_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
):
    random.shuffle(data)
    total = len(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": data[:train_end],
        "val": data[train_end:val_end],
        "test": data[val_end:],
    }

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save standard automodel formats
    for split_name, split_data in splits.items():
        with open(out_path / f"{split_name}.json", "w", encoding="utf-8") as f:
            json.dump({"corpus": {"path": "./corpus"}, "data": split_data}, f, indent=2)

    # Also save NVIDIA BEIR formats for metrics parsing
    beir_dir = out_path / "eval_beir"
    beir_dir.mkdir(parents=True, exist_ok=True)

    q_lines, c_lines, r_lines = [], [], []
    for row in splits["test"]:
        qid = row["question_id"]
        q_lines.append(
            json.dumps(
                {
                    "_id": qid,
                    "text": row["question"],
                    "metadata": {"corpus_id": row["corpus_id"]},
                }
            )
        )
        for pd in row["pos_doc"]:
            pid = pd["id"]
            c_lines.append(json.dumps({"_id": pid, "text": pd["text"]}))
            r_lines.append(f"{qid}\t{pid}\t1")

    with open(beir_dir / "queries.jsonl", "w", encoding="utf-8") as f:
        f.write("\n".join(q_lines))
    # Note: For Beir corpus, it should ideally have deduplicated texts, but simple writing works for now.
    with open(beir_dir / "corpus.jsonl", "w", encoding="utf-8") as f:
        f.write("\n".join(list(set(c_lines))))
    with open(beir_dir / "test.tsv", "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n" + "\n".join(r_lines))

    print(f"Data saved to {out_path}.")
    print(
        f"Splits: Train {len(splits['train'])} / Val {len(splits['val'])} / Test {len(splits['test'])}"
    )


def main(sdg_input: str, output_dir: str, config_path: str, quality_threshold: float):
    print("Loading config...")
    global_cfg = files.read_toml(config_path)
    embed_cfg_dict = global_cfg.get("embedding", {})
    embed_cfg = EmbedConfig(**embed_cfg_dict)

    # Instantiate Embedder
    try:
        from aisa.gen.providers import EmbedInfo, Provider

        embedder = Embedder(
            embed_cfg=embed_cfg, out_dir="./output/embed/stage1_data_prep"
        )
    except Exception as e:
        print(f"Error initializing embedder: {e}")
        return

    print("Step 1: Convert to Retriever JSON")
    converted_data = filter_and_convert(sdg_input, quality_threshold)
    print(f"Valid pairs extracted: {len(converted_data)}")

    print("Step 2: Hard Negative Mining")
    mined_data = mine_hard_negatives(
        converted_data, embedder, top_k=5, hard_neg_margin=0.95
    )

    print("Step 3: Unroll Positive Docs")
    unrolled_data = unroll_pos_docs(mined_data)
    print(f"Total pairs after unrolling: {len(unrolled_data)}")

    print("Step 4: Train / Val / Test Splits")
    save_splits(unrolled_data, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./example-config.toml")
    parser.add_argument(
        "--sdg-input", type=str, required=True, help="Path to generated SDG JSON"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./output/embed/stage1_data_prep"
    )
    parser.add_argument(
        "--threshold", type=float, default=7.0, help="Min quality score"
    )
    args = parser.parse_args()

    main(args.sdg_input, args.output_dir, args.config, args.threshold)
