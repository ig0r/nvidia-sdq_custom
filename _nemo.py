import os
import re
import json
import asyncio
import argparse
import random
from typing import Literal
import numpy as np
import tiktoken as tikt
from pathlib import Path
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from aisa.utils import files, logger, dictlist
from aisa.gen import BaseLLM, LLMConfig, Embedder, EmbedConfig
from aisa.parse.chunk import RecursiveChunker
from aisa.parse.chunkers import Chunker, get_chunker, group_kept_pieces

SYS_PROMPTS: dict[str, str] = {
    "artifacts": "You are an expert at analyzing documents and extracting semantic artifacts.",
    "qa-gen": "You are an expert at extracting question and answer pairs from provided context/transcript/segments.",
    "eval": "You are an expert evaluator of question-answer pairs.",
}

MD_PATTERNS: dict[str, re.Pattern] = {
    "table": re.compile(
        r"(?:\|.*\|.*\n)"  # Header row
        r"(?:\|[\s:\-]+\|.*\n)"  # Separator row
        r"(?:\|.*\|.*\n?)+",  # Data rows
        flags=re.MULTILINE,
    ),
    "image": re.compile(r"!\[.*?\]\((.*?)\)"),
}

ARTIFACT_CATS: list[str] = [
    "key_concepts",
    "relationships",
    "themes",
    "entities",
    "processes",
    "insights",
    "technical_terms",
    "contextual_factors",
]


def get_token_count(text: str) -> int:
    encoding: tikt.Encoding = tikt.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text, disallowed_special=()))


def get_fact_blocks(artifacts: dictlist) -> list[str]:
    blocks: list[str] = []
    for art in artifacts:
        block: str = ""
        for cat in ARTIFACT_CATS:
            if cat in art and art[cat]:
                block += f"<{cat}>\n"
                for item in art[cat]:
                    block += (
                        f"- {item.get('text', '')}: {item.get('description', '')}\n"
                    )
                block += f"</{cat}>\n"
        blocks.append(block.strip())
    return blocks


def get_ctx_blocks(ctx_path) -> list[str]:
    blocks: list[str] = []
    if Path(ctx_path).exists():
        ctx_data: dictlist = files.read_json(ctx_path)
        for entry in ctx_data:
            segments: list[str] = [
                f"Segment {subch['chunk_id']}: {subch['text']}"
                for subch in entry["chunks"]
            ]
            blocks.append("=== Section 1 ===\n" + "\n".join(segments))
        return blocks
    raise FileNotFoundError(f"Context file not found: {ctx_path}")


def _shared_suffix_prefix_len(a: str, b: str, max_scan: int = 1200) -> int:
    max_len = min(len(a), len(b), max_scan)
    for n in range(max_len, 0, -1):
        if a[-n:] == b[:n]:
            return n
    return 0


def _trim_overlap_for_context(chunk_group: dictlist) -> dictlist:
    if not chunk_group:
        return []

    cleaned: dictlist = [dict(chunk_group[0])]
    for subch in chunk_group[1:]:
        cur: dict = dict(subch)
        prev_text: str = cleaned[-1].get("text", "")
        cur_text: str = cur.get("text", "")
        overlap_len: int = _shared_suffix_prefix_len(prev_text, cur_text)
        if overlap_len > 0:
            cur_text = cur_text[overlap_len:].lstrip()
        cur["text"] = cur_text
        cur["tokens"] = get_token_count(cur_text)
        cleaned.append(cur)

    return cleaned


RELEVANCE_SCORE = Literal[0, 0.5, 1]


class RelevanceJudgment(BaseModel):
    score: RELEVANCE_SCORE = Field(
        description="1=clearly relevant pavement engineering content; 0.5=unsure/mixed; 0=closed-list noise"
    )
    reason: str = Field(description="Brief explanation, ≤15 words")


_JSON_BLOCK_RE = re.compile(r"<json>\s*(.*?)\s*</json>", re.DOTALL)
_SCRATCHPAD_BLOCK_RE = re.compile(r"<scratchpad>\s*(.*?)\s*</scratchpad>", re.DOTALL)
_FENCE_OPEN_RE = re.compile(r"\A```\w*\s*", re.MULTILINE)
_FENCE_CLOSE_RE = re.compile(r"\s*```\s*\Z", re.MULTILINE)


class QAGenerator:
    def __init__(
        self,
        root_dir: str,
        input_dir: str = "./test_md",
        llm: BaseLLM = BaseLLM(LLMConfig()),
        embedder: Embedder = Embedder(EmbedConfig()),
        chunk_cfg: dict = None,
    ):
        self.llm: BaseLLM = llm
        self.embedder: Embedder = embedder

        # hard codes
        self.overwrite: bool = False
        self.max_artifacts: int = 2
        self.query_counts_multi_hop: int = 5
        self.query_counts_structural: int = 5
        self.query_counts_contextual: int = 5
        self.reasoning_counts_factual: int = 3
        self.reasoning_counts_relational: int = 3
        self.reasoning_counts_inferential: int = 3
        self.reasoning_counts_temporal: int = 3
        self.reasoning_counts_procedural: int = 3
        self.reasoning_counts_visual: int = 0
        self.reasoning_counts_causal: int = 0
        self.min_hops: int = 2
        self.max_hops: int = 3
        self.min_complexity: int = 3
        self.num_pairs: int = 15

        # chunker (backward-compatible: missing [chunking] falls back to recursive w/ embedding params)
        if not chunk_cfg:
            chunk_cfg = {
                "method": "recursive",
                "chunk_size": embedder.cfg.chunk_size if embedder.cfg else 256,
                "recursive_overlap": embedder.cfg.recursive_overlap if embedder.cfg else 50,
            }
        self.chunk_cfg: dict = chunk_cfg
        self.chunker: Chunker = get_chunker(chunk_cfg, llm)

        # relevance filter (mode 3 only). Eager AsyncOpenAI init when enabled.
        self.relevance_concurrency: int = max(
            1, int(chunk_cfg.get("relevance_concurrency", 8))
        )
        self.eval_client: AsyncOpenAI | None = None
        filter_on: bool = bool(chunk_cfg.get("relevance_filter", False))
        if filter_on and chunk_cfg.get("method") == "random_logical":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is required when [chunking].relevance_filter = true"
                )
            self.eval_client = AsyncOpenAI(api_key=api_key)
        elif filter_on:
            logger.log(
                "CHUNK",
                f"relevance_filter ignored: only honored for method='random_logical' "
                f"(got method={chunk_cfg.get('method')!r})",
            )

        # build directories
        method: str = chunk_cfg.get("method", "recursive")
        c_size: int = chunk_cfg.get("chunk_size", 256)
        self.root_dir: str = root_dir
        self.input_dir: str = input_dir
        self.chunk_dir: str = files.append_directory(
            root_dir, f"doc-chunks_{c_size}_{method}"
        )
        self.doc_paths: dict[Path, str] = {}

        # main task mapping
        self.tasks: dict[str, callable] = {
            "chunk": self.run_chunk_only_pipeline,
            "sdg": self.run_sgd_pipeline,
            "sdg_logical": self.run_sgd_logical_pipeline,
            "prep": self.run_data_prep_pipeline,
        }

    async def run_chunk_only_pipeline(self):
        md_files: list[Path] = sorted(Path(self.input_dir).glob("*.md"))
        if not md_files:
            logger.log("NLP", f"No .md files found in {self.input_dir}")
            return
        for file_path in md_files:
            chunks: dictlist = await self.path2chunks(file_path)
            logger.log(
                "CHUNK",
                f"{file_path.name}: {len(chunks)} chunks -> {self.doc_paths[file_path]}",
            )

    async def path2chunks(self, file_path: Path) -> dictlist:
        abs_path, filename = files.split_path(str(file_path))
        doc_id: str = "_".join(filename.split("_")[:2])
        base_out: str = f"{self.chunk_dir}/{doc_id}-chunks.json"
        self.doc_paths[file_path] = base_out
        method: str = self.chunk_cfg.get("method", "recursive")
        is_hybrid: bool = method == "random_logical"
        cache_path: str = (
            f"{self.chunk_dir}/{doc_id}-logic-chunks.json" if is_hybrid else base_out
        )

        if Path(cache_path).exists() and not self.overwrite:
            return files.read_json(cache_path).get("texts", [])

        raw_text = file_path.read_text(encoding="utf-8")
        tables = MD_PATTERNS["table"].findall(raw_text)
        images = MD_PATTERNS["image"].findall(raw_text)
        raw_text = MD_PATTERNS["table"].sub("", raw_text)
        raw_text = MD_PATTERNS["image"].sub("", raw_text)
        parsed_file: str = str(abs_path) + "/" + filename

        if is_hybrid:
            filter_on: bool = self.eval_client is not None

            if filter_on:
                # 1. Recursive pre-split (driven inline so we can filter before grouping)
                rec_pieces: list[str] = self.chunker.recursive.split(raw_text)
            else:
                # Existing flow: chunker.split() does both pre-split and grouping in one pass
                raw_chunks_text: list[str] = self.chunker.split(raw_text)
                rec_pieces = self.chunker.last_recursive_pieces

            rec_chunks: dictlist = [
                {
                    "text": p,
                    "chunk_id": idx,
                    "u_chunk_id": f"{doc_id}-chunk-{idx}",
                    "tokens": get_token_count(p),
                }
                for idx, p in enumerate(rec_pieces)
            ]
            files.write_json(
                {
                    "doc_id": doc_id,
                    "parsed_file": parsed_file,
                    "texts": rec_chunks,
                    "images": images,
                    "tables": tables,
                },
                base_out,
            )

            if filter_on:
                # 2. Per-piece relevance evaluation
                try:
                    scores: dictlist = await self.evaluate_chunks(
                        file_path, rec_chunks
                    )
                    kept_indices: list[int] = [
                        c["chunk_id"]
                        for c, s in zip(rec_chunks, scores)
                        if s["score"] > 0.5
                    ]
                except Exception as exc:
                    logger.log(
                        "CHUNK",
                        f"{file_path.name}: relevance filter failed ({exc!r}); "
                        f"falling back to keep-all",
                    )
                    kept_indices = list(range(len(rec_pieces)))

                # 3. Mask-aware logical grouping
                raw_chunks_text, sources = group_kept_pieces(
                    rec_pieces,
                    kept_indices,
                    self.llm,
                    self.chunker.prompt_template,
                    self.chunker.window,
                    self.chunker.stride,
                    self.chunker.recursive_overlap > 0,
                )
            else:
                sources = self.chunker.last_source_indices

            logic_chunks: dictlist = [
                {
                    "text": ch,
                    "chunk_id": idx,
                    "u_logic_chunk_id": f"{doc_id}-logic-chunk-{idx}",
                    "tokens": get_token_count(ch),
                    "source_chunk_ids": sources[idx] if idx < len(sources) else [],
                    "source_u_chunk_ids": [
                        f"{doc_id}-chunk-{src}"
                        for src in (sources[idx] if idx < len(sources) else [])
                    ],
                }
                for idx, ch in enumerate(raw_chunks_text)
            ]
            files.write_json(
                {
                    "doc_id": doc_id,
                    "parsed_file": parsed_file,
                    "texts": logic_chunks,
                    "images": images,
                    "tables": tables,
                },
                cache_path,
            )
            return logic_chunks

        # Non-hybrid (mode 1, mode 2): unchanged
        raw_chunks_text = self.chunker.split(raw_text)
        chunks: dictlist = [
            {"text": ch, "chunk_id": idx, "tokens": get_token_count(ch)}
            for idx, ch in enumerate(raw_chunks_text)
        ]
        files.write_json(
            {
                "doc_id": doc_id,
                "parsed_file": parsed_file,
                "texts": chunks,
                "images": images,
                "tables": tables,
            },
            base_out,
        )

        return chunks

    async def evaluate_chunks(self, file_path: Path, chunks: dictlist) -> dictlist:
        """Score each recursive chunk for pavement-engineering relevance via OpenAI
        chat completions with chain-of-thought + tagged-JSON output. One LLM call per
        recursive piece; calls run concurrently bounded by self.relevance_concurrency.

        Caller (path2chunks) only invokes this when method == "random_logical" and
        [chunking].relevance_filter is true; the method itself does not gate on mode.
        Assumes self.eval_client is not None (constructed in __init__ when the filter
        is on).
        """
        base_out: str = self.doc_paths[file_path].replace(
            "-chunks.json", "-relevance.json"
        )
        if Path(base_out).exists() and not self.overwrite:
            cached = files.read_json(base_out)
            scores_cached: dictlist = (
                cached.get("scores", []) if isinstance(cached, dict) else cached
            )
            logger.log("CHUNK", f"{file_path.name}: cache hit -> {base_out}")
            return scores_cached

        prompt_template: str = self.llm.read_prompt("nemo_eval-02")
        sem = asyncio.Semaphore(self.relevance_concurrency)

        async def _eval_one(chunk: dict) -> dict:
            cid: int = chunk["chunk_id"]
            u_cid: str = chunk["u_chunk_id"]
            async with sem:
                try:
                    user_content = prompt_template.format(CHUNK=chunk["text"])
                    completion = await self.eval_client.chat.completions.create(
                        model="gpt-4o-mini",
                        temperature=0.0,
                        messages=[{"role": "user", "content": user_content}],
                    )
                    text: str = completion.choices[0].message.content or ""

                    json_match = _JSON_BLOCK_RE.search(text)
                    if not json_match:
                        raise RuntimeError(
                            f"no <json> block in response (head: {text[:200]!r})"
                        )
                    json_text = json_match.group(1).strip()
                    json_text = _FENCE_OPEN_RE.sub("", json_text)
                    json_text = _FENCE_CLOSE_RE.sub("", json_text).strip()

                    judgment = RelevanceJudgment.model_validate_json(json_text)

                    scratch_match = _SCRATCHPAD_BLOCK_RE.search(text)
                    scratchpad = (
                        scratch_match.group(1).strip() if scratch_match else None
                    )

                    return {
                        "chunk_id": cid,
                        "u_chunk_id": u_cid,
                        "score": float(judgment.score),
                        "reason": judgment.reason,
                        "scratchpad": scratchpad,
                    }
                except Exception as exc:
                    logger.log(
                        "CHUNK",
                        f"{file_path.name}: chunk_id={cid} eval failed "
                        f"({exc!r}); defaulting to score=1.0",
                    )
                    return {
                        "chunk_id": cid,
                        "u_chunk_id": u_cid,
                        "score": 1.0,
                        "reason": f"error: {exc}",
                        "scratchpad": None,
                    }

        scores: dictlist = await asyncio.gather(*[_eval_one(c) for c in chunks])

        bins: dict[float, int] = {0.0: 0, 0.5: 0, 1.0: 0}
        for s in scores:
            bins[s["score"]] = bins.get(s["score"], 0) + 1
        logger.log(
            "CHUNK",
            f"{file_path.name}: {bins[1.0]}/{len(chunks)} pieces kept "
            f"(filtered {bins[0.0]}; unsure {bins[0.5]})",
        )

        doc_id: str = Path(self.doc_paths[file_path]).name.replace(
            "-chunks.json", ""
        )
        files.write_json({"doc_id": doc_id, "scores": scores}, base_out)
        return scores

    async def extract_artifacts(self, file_path: Path, chunks: dictlist) -> dictlist:
        prompt: str = self.llm.read_prompt("nemo_artifacts")
        base_out: str = self.doc_paths[file_path].replace(
            "-chunks.json", "-artifacts.json"
        )
        ctx: dictlist = []
        res: dictlist = []

        if Path(base_out).exists() and not self.overwrite:
            return files.read_json(base_out)

        new_chunks = RecursiveChunker(
            custom_input=chunks,
            prompt=prompt.replace("{max_artifacts}", str(self.max_artifacts)),
            max_input_tokens=self.llm.cfg.max_input_tokens,  #  + 2000,
            return_type=list,
        )
        # Only de-overlap local context bundles for artifact extraction.
        ctx_chunks: list[dictlist] = [_trim_overlap_for_context(c) for c in new_chunks]
        ctx = [
            {
                "chunks": c,
                "tokens": sum(subch.get("tokens", 0) for subch in c),
            }
            for c in ctx_chunks
        ]

        files.write_json(
            ctx,
            self.doc_paths[file_path].replace("-chunks.json", "-ctx.json"),
        )
        res = await self.llm.run_chain(
            prompt,
            [
                {
                    "text": "\n".join([subch.get("text", "") for subch in c]),
                    "max_artifacts": self.max_artifacts,
                }
                for c in ctx_chunks
            ],
        )
        files.write_json(res, base_out)
        return res

    async def generate_qa_pairs(
        self,
        file_path: Path,
        fact_blocks: list[str],
        ctx_blocks: list[str],
    ) -> dictlist:
        prompt: str = self.llm.read_prompt("nemo_qa-gen")
        base_out: str = self.doc_paths[file_path].replace(
            "-chunks.json", "-qa_pairs.json"
        )
        res: dictlist = []

        if Path(base_out).exists() and not self.overwrite:
            return files.read_json(base_out)

        res = await self.llm.run_chain(
            prompt,
            [
                {
                    "facts_block": block,
                    "context_block": ctx_blocks[idx],
                    "query_counts_multi_hop": self.query_counts_multi_hop,
                    "query_counts_structural": self.query_counts_structural,
                    "query_counts_contextual": self.query_counts_contextual,
                    "reasoning_counts_factual": self.reasoning_counts_factual,
                    "reasoning_counts_relational": self.reasoning_counts_relational,
                    "reasoning_counts_inferential": self.reasoning_counts_inferential,
                    "reasoning_counts_temporal": self.reasoning_counts_temporal,
                    "reasoning_counts_procedural": self.reasoning_counts_procedural,
                    "reasoning_counts_visual": self.reasoning_counts_visual,
                    "reasoning_counts_causal": self.reasoning_counts_causal,
                    "min_hops": self.min_hops,
                    "max_hops": self.max_hops,
                    "min_complexity": self.min_complexity,
                    "num_pairs": self.num_pairs,
                }
                for idx, block in enumerate(fact_blocks)
            ],
        )
        files.write_json(res, base_out)
        return res

    async def evaluate_qa_pairs(
        self,
        file_path: Path,
        ctx_blocks: list[str],
        qa_pairs: dictlist,
    ) -> dictlist:
        qa_blocks: list[str] = []
        cnt: int = 1
        prompt: str = self.llm.read_prompt("nemo_eval")
        base_out: str = self.doc_paths[file_path].replace(
            "-chunks.json", "-qa_eval.json"
        )

        if Path(base_out).exists() and not self.overwrite:
            return files.read_json(base_out)

        for qas in qa_pairs:
            qa_block: str = ""
            for pair in qas["pairs"]:
                qa_block += f"=== QA Pair {cnt} ===\n\n"
                qa_block += f"QUESTION: {pair.get('question')}\n\n"
                qa_block += f"ANSWER: {pair.get('answer')}\n\n"
                qa_block += (
                    f"CONTEXT: (Relevant Segment IDs): {pair.get('segment_ids')}\n\n"
                )
                cnt += 1
            qa_blocks.append(qa_block)

        res = await self.llm.run_chain(
            prompt,
            [
                {
                    "qa_pairs_block": qa_blocks[idx],
                    "segments_block": ctx_blocks[idx],
                    "num_pairs": len(qa_blocks[idx].split("=== QA Pair")) - 1,
                }
                for idx in range(len(qa_blocks))
            ],
        )
        files.write_json(res, base_out)
        return res

    async def run_sgd_pipeline(self):
        res: dictlist = []
        for file_path in Path(self.input_dir).glob("*.md"):
            chunks: dictlist = await self.path2chunks(file_path)
            artifacts: dictlist = await self.extract_artifacts(file_path, chunks)
            fact_blocks: list[str] = get_fact_blocks(artifacts)
            ctx_blocks: list[str] = get_ctx_blocks(
                self.doc_paths[file_path].replace("-chunks.json", "-ctx.json")
            )
            qa_pairs: dictlist = await self.generate_qa_pairs(
                file_path, fact_blocks, ctx_blocks
            )
            evals: dictlist = await self.evaluate_qa_pairs(
                file_path, ctx_blocks, qa_pairs
            )
            res_entry = {
                "file_name": file_path.name,
                # "text": "\n".join([ch["text"] for ch in chunks]),
                "chunks": chunks,
                "qa_pairs": [],
                "qa_evals": [],
            }
            for idx in range(len(qa_pairs)):
                res_entry["qa_pairs"].extend(qa_pairs[idx].get("pairs", []))
                res_entry["qa_evals"].extend(evals[idx].get("evaluations", []))
            res.append(res_entry)
            files.write_json(
                res_entry,
                self.doc_paths[file_path].replace("-chunks.json", "-sdg.json"),
            )
        files.write_json(res, f"{self.root_dir}/full_sdg_output.json")

    def _build_logical_contexts(self, file_path: Path, chunks: dictlist) -> dictlist:
        base_path: str = self.doc_paths[file_path]
        out_path: str = base_path.replace("-chunks.json", "-logic-ctx.json")
        doc_id: str = Path(base_path).name.replace("-chunks.json", "")

        if Path(out_path).exists() and not self.overwrite:
            logger.log("CHUNK", f"{file_path.name}: cache hit -> {out_path}")
            cached = files.read_json(out_path)
            return cached.get("contexts", []) if isinstance(cached, dict) else cached

        def _ctx_chunk(chunk: dict) -> dict:
            out: dict = {}
            for k, v in chunk.items():
                if k == "source_u_chunk_ids":
                    out["source_u_logic_chunk_ids"] = [chunk.get("u_logic_chunk_id")]
                out[k] = v
            if "source_u_logic_chunk_ids" not in out:
                out["source_u_logic_chunk_ids"] = [chunk.get("u_logic_chunk_id")]
            return out

        ctx: dictlist = [
            {
                "u_ctx_id": f"{doc_id}-ctx-{idx}",
                "chunks": [_ctx_chunk(chunk)],
                "tokens": chunk.get("tokens", 0),
            }
            for idx, chunk in enumerate(chunks)
        ]

        budget: int = self.llm.cfg.max_input_tokens
        for entry in ctx:
            if entry["tokens"] > budget:
                cid = entry["chunks"][0].get("chunk_id")
                logger.log(
                    "CHUNK",
                    f"{file_path.name}: chunk_id={cid} tokens={entry['tokens']} > max_input_tokens={budget}",
                )

        files.write_json({"doc_id": doc_id, "contexts": ctx}, out_path)
        return ctx

    async def run_sgd_logical_pipeline(self):
        method: str = self.chunk_cfg.get("method")
        if method != "random_logical":
            raise ValueError(
                f"--sdg-logical requires [chunking].method == 'random_logical'; got {method!r}. "
                f"For mode 'logical', use --sdg instead."
            )

        md_files: list[Path] = sorted(Path(self.input_dir).glob("*.md"))
        if not md_files:
            logger.log("NLP", f"No .md files found in {self.input_dir}")
            return

        for file_path in md_files:
            chunks: dictlist = await self.path2chunks(file_path)
            ctx: dictlist = self._build_logical_contexts(file_path, chunks)
            out_path: str = self.doc_paths[file_path].replace(
                "-chunks.json", "-logic-ctx.json"
            )
            logger.log(
                "CHUNK",
                f"{file_path.name}: {len(ctx)} logical-context entries -> {out_path}",
            )

    def filter_and_convert(self, sdg_records: dictlist, quality_threshold: float = 7.0) -> dictlist:
        training_docs = []
        question_counter = 0
        for record in sdg_records:
            file_name = record.get("file_name", "unknown")
            chunks = {chunk["chunk_id"]: chunk["text"] for chunk in record.get("chunks", [])}
            qa_pairs = record.get("qa_pairs", [])
            evaluations = record.get("qa_evals", [])
            for i, qa in enumerate(qa_pairs):
                score = 0.0
                if i < len(evaluations):
                    eval_data = evaluations[i]
                    overall = eval_data.get("overall", {})
                    score = overall.get("score", 0.0)
                if score < quality_threshold:
                    continue
                segment_ids = qa.get("segment_ids", [])
                pos_docs = []
                for sid in segment_ids:
                    if sid in chunks:
                        pos_docs.append({
                            "id": f"{file_name}_chunk_{sid}",
                            "text": chunks[sid]
                        })
                if not pos_docs:
                    continue
                q_id = f"q{question_counter:06d}"
                question_counter += 1
                training_docs.append({
                    "question_id": q_id,
                    "question": qa.get("question", ""),
                    "corpus_id": file_name,
                    "pos_doc": pos_docs,
                    "neg_doc": []
                })
        return training_docs

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        return np.dot(a_norm, b_norm.T)

    async def mine_hard_negatives(self, data: dictlist, top_k: int = 5, hard_neg_margin: float = 0.95) -> dictlist:
        logger.log("NLP", "Extracting unique passages for embeddings cache...")
        passage_text_to_id = {}
        for row in data:
            for pd in row.get("pos_doc", []):
                p_text = pd["text"]
                p_id = pd["id"]
                if p_text not in passage_text_to_id:
                    passage_text_to_id[p_text] = p_id
        unique_texts = list(passage_text_to_id.keys())
        if not unique_texts:
            logger.log("NLP", "No valid passages found! Cannot mine hard negatives.")
            return data
            
        passage_embeddings = await self.embedder.embed_docs(unique_texts, verbose=False)
        queries_text = [row["question"] for row in data]
        query_embeddings = await self.embedder.embed_docs(queries_text, verbose=False)
        sim_matrix = self.cosine_similarity(query_embeddings, passage_embeddings)
        
        for i, row in enumerate(data):
            target_ids = {pd["id"] for pd in row["pos_doc"]}
            sims = sim_matrix[i]
            top_indices = np.argsort(sims)[::-1]
            hard_negs = []
            for idx in top_indices:
                p_text = unique_texts[idx]
                p_id = passage_text_to_id[p_text]
                score = sims[idx]
                if p_id in target_ids:
                    continue
                if score <= hard_neg_margin:
                    hard_negs.append({"id": p_id, "text": p_text, "score": float(score)})
                if len(hard_negs) >= top_k:
                    break
            row["neg_doc"] = hard_negs
        return data

    def unroll_pos_docs(self, data: dictlist) -> dictlist:
        logger.log("NLP", "Unrolling multi-pos documents...")
        unrolled = []
        for record in data:
            pos_docs = record.get("pos_doc", [])
            if len(pos_docs) <= 1:
                unrolled.append(record)
            else:
                base_q_id = record["question_id"]
                for idx, pd in enumerate(pos_docs):
                    unrolled.append({
                        "question_id": f"{base_q_id}_{idx}",
                        "question": record["question"],
                        "corpus_id": record["corpus_id"],
                        "pos_doc": [pd],
                        "neg_doc": record.get("neg_doc", [])
                    })
        return unrolled

    def save_splits(self, data: dictlist, train_ratio: float = 0.8, val_ratio: float = 0.1):
        random.shuffle(data)
        total = len(data)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        splits = {
            "train": data[:train_end],
            "val": data[train_end:val_end],
            "test": data[val_end:]
        }
        
        out_path = Path(self.root_dir) / "embed_data_prep"
        out_path.mkdir(parents=True, exist_ok=True)
        
        for split_name, split_data in splits.items():
            files.write_json({
                "corpus": {"path": "./corpus"},
                "data": split_data
            }, str(out_path / f"{split_name}.json"))
            
        beir_dir = out_path / "eval_beir"
        beir_dir.mkdir(parents=True, exist_ok=True)
        
        q_lines, c_lines, r_lines = [], [], []
        for row in splits["test"]:
            qid = row["question_id"]
            q_lines.append(json.dumps({
                "_id": qid,
                "text": row["question"],
                "metadata": {"corpus_id": row["corpus_id"]}
            }))
            for pd in row["pos_doc"]:
                pid = pd["id"]
                c_lines.append(json.dumps({"_id": pid, "text": pd["text"]}))
                r_lines.append(f"{qid}\t{pid}\t1")
                
        with open(beir_dir / "queries.jsonl", "w", encoding="utf-8") as f:
            f.write("\n".join(q_lines))
        with open(beir_dir / "corpus.jsonl", "w", encoding="utf-8") as f:
            f.write("\n".join(list(set(c_lines))))
        with open(beir_dir / "test.tsv", "w", encoding="utf-8") as f:
            f.write("query-id\tcorpus-id\tscore\n" + "\n".join(r_lines))
            
        logger.info(f"Splits: Train {len(splits['train'])} / Val {len(splits['val'])} / Test {len(splits['test'])}")

    async def run_data_prep_pipeline(self, quality_threshold: float = 7.0):
        sdg_output_path = f"{self.root_dir}/full_sdg_output.json"
        if not Path(sdg_output_path).exists():
            logger.log("NLP", f"Error: SDG output not found at {sdg_output_path}")
            return
            
        sdg_records = files.read_json(sdg_output_path)
        converted_data = self.filter_and_convert(sdg_records, quality_threshold)
        mined_data = await self.mine_hard_negatives(converted_data, top_k=5, hard_neg_margin=0.95)
        unrolled_data = self.unroll_pos_docs(mined_data)
        self.save_splits(unrolled_data)


async def main(cfg: dict):
    qagen: QAGenerator = QAGenerator(
        root_dir=cfg["general"]["output_dir"],
        input_dir=cfg["general"]["data_dir"],
        llm=BaseLLM(LLMConfig(**cfg["llm"])),
        embedder=Embedder(EmbedConfig(**cfg["embedding"])),
        chunk_cfg=cfg.get("chunking"),
    )

    for task_name, should_run in cfg["nemo_task"].items():
        if should_run and task_name in qagen.tasks:
            logger.log("NLP", f"Starting task: {task_name}")
            await qagen.tasks[task_name]()
        else:
            logger.log("NLP", f"Skipping task: {task_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-only", action="store_true", help="Run chunking only (stops after path2chunks)")
    parser.add_argument("--sdg", action="store_true", help="Run SDG")
    parser.add_argument("--sdg-logical", action="store_true", help="Run SDG on logical chunks (Step 1: bundle only)")
    parser.add_argument("--prep", action="store_true", help="Run data prep")
    parser.add_argument("--cfg", type=str, default="./cfg/nemo.toml", help="Cfg")
    parser.add_argument("--input_dir", type=str, help="Input directory")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    args = parser.parse_args()

    global_cfg: dict = files.read_toml(args.cfg)
    if args.output_dir:
        global_cfg["general"]["output_dir"] = args.output_dir
    if args.input_dir:
        global_cfg["general"]["data_dir"] = args.input_dir
    global_cfg["nemo_task"] = {
        "chunk": args.chunk_only,
        "sdg": args.sdg,
        "sdg_logical": args.sdg_logical,
        "prep": args.prep,
    }
    if not any(global_cfg["nemo_task"].values()):
        parser.error("no task selected: pass at least one of --chunk-only, --sdg, --sdg-logical, --prep")
    files.create_folder(global_cfg["general"]["output_dir"])
    asyncio.run(main(global_cfg))
