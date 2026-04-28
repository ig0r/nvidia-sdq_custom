#!/usr/bin/env python3
"""Standalone CLI: extract pavement-engineering artifacts from mode-3 logical chunks.

Reads {output_dir}/doc-chunks_{size}_random_logical/*-logic-chunks.json, runs
langextract per logical chunk with OpenAI gpt-4o-mini, writes *-logic-artifacts.json.

Requires:
- [chunking].method == 'random_logical' in the cfg TOML.
- OPENAI_API_KEY in env or .env.
- pip install langextract loguru python-dotenv

Single-file standalone: no imports from the project's aisa/ package.
"""
import argparse
import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
import langextract as lx


load_dotenv()

# Custom log levels matching aisa/utils/log.py so output style stays consistent
# with the rest of the project. logger.level() raises if the level already exists,
# which can happen if this module is imported alongside the aisa logger.
for _name, _no, _icon in (("CHUNK", 14, "📦"), ("NLP", 10, "🧠")):
    try:
        logger.level(_name, no=_no, icon=_icon)
    except (TypeError, ValueError):
        pass


EXTRACTION_CLASSES: list[str] = [
    "requirement",
    "condition",
    "exception",
    "constraint",
    "procedure",
    "method",
    "formula",
    "parameter",
    "threshold",
    "definition",
    "actor_role",
    "deliverable",
    "assumption",
    "finding",
    "recommendation",
    "best_practice",
    "decision",
    "rationale",
    "issue",
    "risk",
    "evidence",
]


PAVEMENT_EXAMPLES: list[lx.data.ExampleData] = [
    lx.data.ExampleData(
        text=(
            "If the projected 18-kip ESALs vary substantially between adjacent "
            "construction sections, separate pavement design analyses shall be "
            "prepared. The designer shall submit the pavement design analysis "
            "to the District for approval. Good pavement design practice is to "
            "provide adequate drainage to prevent water from being trapped within "
            "the pavement structure, which can reduce pavement performance."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="condition",
                extraction_text=(
                    "If the projected 18-kip ESALs vary substantially between "
                    "adjacent construction sections"
                ),
                attributes={
                    "trigger": "projected 18-kip ESALs vary substantially",
                    "applies_to": "adjacent construction sections",
                    "source_cue": "If",
                    "description": "condition under which separate design analyses are needed",
                },
            ),
            lx.data.Extraction(
                extraction_class="requirement",
                extraction_text="separate pavement design analyses shall be prepared",
                attributes={
                    "modality": "shall",
                    "required_action": "prepared",
                    "target": "separate pavement design analyses",
                    "condition": (
                        "projected 18-kip ESALs vary substantially between "
                        "adjacent construction sections"
                    ),
                    "source_cue": "shall",
                    "description": "requirement to prepare separate pavement design analyses",
                },
            ),
            lx.data.Extraction(
                extraction_class="actor_role",
                extraction_text="The designer",
                attributes={
                    "role": "designer",
                    "responsibility": "submit the pavement design analysis",
                    "description": "party responsible for submitting the pavement design analysis",
                },
            ),
            lx.data.Extraction(
                extraction_class="requirement",
                extraction_text=(
                    "The designer shall submit the pavement design analysis "
                    "to the District for approval"
                ),
                attributes={
                    "modality": "shall",
                    "actor": "designer",
                    "required_action": "submit",
                    "target": "pavement design analysis",
                    "recipient": "District",
                    "purpose": "approval",
                    "source_cue": "shall",
                    "description": "requirement for the designer to submit the design analysis for District approval",
                    "significance": "submission to the District is required for design approval",
                },
            ),
            lx.data.Extraction(
                extraction_class="deliverable",
                extraction_text="pavement design analysis",
                attributes={
                    "name": "pavement design analysis",
                    "deliverable_type": "analysis",
                    "recipient": "District",
                    "purpose": "approval",
                    "description": "analysis submitted for pavement design approval",
                    "significance": "submitted to the District for approval",
                },
            ),
            lx.data.Extraction(
                extraction_class="best_practice",
                extraction_text=(
                    "Good pavement design practice is to provide adequate drainage "
                    "to prevent water from being trapped within the pavement structure"
                ),
                attributes={
                    "practice": "provide adequate drainage",
                    "target": "pavement structure",
                    "purpose": "prevent water from being trapped",
                    "basis": "good pavement design practice",
                    "source_cue": "Good pavement design practice",
                    "description": "preferred drainage practice for pavement design",
                    "significance": "trapped water can reduce pavement performance",
                },
            ),
        ],
    ),
]


@dataclass
class LXConfig:
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    temperature: float = 0.0
    extraction_passes: int = 2
    max_char_buffer: int = 10000
    prompt_name: str = "nemo_logic-artifacts-02"
    prompt_lib: str = "./prompts"


class PavementExtractor:
    def __init__(self, cfg: LXConfig):
        self.cfg: LXConfig = cfg
        api_key: str | None = cfg.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set (env or [langextract].api_key in cfg)"
            )
        self.api_key: str = api_key
        prompt_path: Path = Path(cfg.prompt_lib) / f"{cfg.prompt_name}.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt not found: {prompt_path}")
        self.prompt: str = prompt_path.read_text(encoding="utf-8")

    def extract(self, text: str, doc_id: str, chunk_id: int) -> dict[str, list[dict]]:
        result = lx.extract(
            text_or_documents=text,
            prompt_description=self.prompt,
            examples=PAVEMENT_EXAMPLES,
            model_id=self.cfg.model,
            api_key=self.api_key,
            temperature=self.cfg.temperature,
            extraction_passes=self.cfg.extraction_passes,
            max_char_buffer=self.cfg.max_char_buffer,
            show_progress=False,
        )
        bucketed: dict[str, list[dict]] = {}
        ext_idx: int = 0
        for ext in (result.extractions or []):
            cls: str = ext.extraction_class
            if cls not in EXTRACTION_CLASSES:
                logger.log(
                    "NLP",
                    f"{doc_id} chunk {chunk_id}: dropping out-of-vocab class {cls!r}",
                )
                continue
            ci = ext.char_interval
            char_iv: dict | None = (
                {"start_pos": ci.start_pos, "end_pos": ci.end_pos}
                if ci is not None
                else None
            )
            attrs: dict = dict(ext.attributes or {})
            description: str = attrs.pop("description", "") or ""
            significance: str | None = attrs.pop("significance", None) or None
            entry: dict = {
                "artifact_id": f"{doc_id}_chunk_{chunk_id}_art_{ext_idx}",
                "text": ext.extraction_text,
                "description": description,
                "significance": significance,
                "char_interval": char_iv,
                "attributes": attrs,
            }
            bucketed.setdefault(cls, []).append(entry)
            ext_idx += 1
        return bucketed


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def _resolve_input_dir(cfg: dict) -> Path:
    """Return the directory holding *-logic-chunks.json. Prefers [paths].input_dir
    (extract_artifacts.toml). For cfg/nemo.toml shape, falls back to globbing
    {[general].output_dir}/doc-chunks_*_random_logical/ and accepting the single
    match — errors if zero or multiple."""
    paths_cfg: dict = cfg.get("paths", {})
    if paths_cfg.get("input_dir"):
        return Path(paths_cfg["input_dir"])

    output_dir: str | None = cfg.get("general", {}).get("output_dir")
    if not output_dir:
        raise ValueError(
            "config missing input_dir: set [paths].input_dir (extract_artifacts.toml) "
            "or [general].output_dir (cfg/nemo.toml)"
        )
    candidates: list[Path] = sorted(Path(output_dir).glob("doc-chunks_*_random_logical"))
    if not candidates:
        raise ValueError(
            f"no doc-chunks_*_random_logical/ directory under {output_dir}; "
            f"run `_nemo.py --chunk-only` with [chunking].method = 'random_logical' first"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"multiple doc-chunks_*_random_logical/ directories under {output_dir}: "
            f"{[c.name for c in candidates]}. Set [paths].input_dir explicitly."
        )
    return candidates[0]


def main(cfg: dict, overwrite: bool = False) -> None:
    # If [chunking].method is present (cfg/nemo.toml shape), enforce mode 3.
    # If absent (extract_artifacts.toml shape), mode 3 is implied.
    method: str = cfg.get("chunking", {}).get("method", "random_logical")
    if method != "random_logical":
        raise ValueError(
            f"extract_artifacts requires [chunking].method == 'random_logical'; got {method!r}"
        )
    chunk_dir: Path = _resolve_input_dir(cfg)
    if not chunk_dir.exists():
        logger.log("NLP", f"Chunk dir not found: {chunk_dir}; nothing to do")
        return

    logic_chunk_files: list[Path] = sorted(chunk_dir.glob("*-logic-chunks.json"))
    if not logic_chunk_files:
        logger.log("NLP", f"No *-logic-chunks.json under {chunk_dir}; nothing to do")
        return

    lx_cfg: LXConfig = LXConfig(**cfg.get("langextract", {}))
    extractor: PavementExtractor = PavementExtractor(lx_cfg)

    for chunks_path in logic_chunk_files:
        doc_id: str = chunks_path.name.replace("-logic-chunks.json", "")
        out_path: Path = chunks_path.with_name(f"{doc_id}-logic-artifacts.json")

        if out_path.exists() and not overwrite:
            logger.log("CHUNK", f"{doc_id}: cache hit -> {out_path}")
            continue

        chunks_doc: dict = _read_json(chunks_path)
        texts: list[dict] = chunks_doc.get("texts", [])

        artifacts: list[dict] = []
        for chunk in texts:
            chunk_id: int = chunk.get("chunk_id")
            tokens: int = chunk.get("tokens", 0)
            try:
                extractions: dict = extractor.extract(
                    text=chunk["text"], doc_id=doc_id, chunk_id=chunk_id
                )
                artifacts.append({
                    "chunk_id": chunk_id,
                    "tokens": tokens,
                    "extractions": extractions,
                })
            except Exception as exc:
                logger.log(
                    "NLP",
                    f"{doc_id} chunk {chunk_id}: extraction failed: {exc}",
                )
                artifacts.append({
                    "chunk_id": chunk_id,
                    "tokens": tokens,
                    "extractions": {},
                    "error": str(exc),
                })

        _write_json({"doc_id": doc_id, "artifacts": artifacts}, out_path)
        logger.log(
            "CHUNK",
            f"{doc_id}: extracted artifacts from {len(artifacts)} logical chunks -> {out_path}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract pavement-engineering artifacts from mode-3 logical chunks."
    )
    parser.add_argument("--cfg", type=str, default="./extract_artifacts.toml",
                        help="Path to TOML config (default: ./extract_artifacts.toml). "
                             "Also accepts cfg/nemo.toml.")
    parser.add_argument("--input_dir", type=str,
                        help="Override the directory containing *-logic-chunks.json")
    parser.add_argument("--overwrite", action="store_true",
                        help="Force regeneration of cached -logic-artifacts.json")
    args = parser.parse_args()

    with open(args.cfg, "rb") as f:
        cfg = tomllib.load(f)
    if args.input_dir:
        cfg.setdefault("paths", {})["input_dir"] = args.input_dir

    # Mode-3 guard mirrors the one in main(): only fires when method is explicitly set.
    method = cfg.get("chunking", {}).get("method", "random_logical")
    if method != "random_logical":
        parser.error(
            f"requires [chunking].method == 'random_logical'; got {method!r}. "
            f"For mode 'logical', use --sdg instead."
        )

    main(cfg, overwrite=args.overwrite)
