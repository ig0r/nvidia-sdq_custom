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
    "material",
    "distress",
    "treatment",
    "specification",
    "test_method",
    "metric",
    "process",
    "reference",
]


PAVEMENT_EXAMPLES: list[lx.data.ExampleData] = [
    lx.data.ExampleData(
        text=(
            "This study evaluated the long-term performance of an ultrathin whitetopping overlay "
            "placed on a deteriorated brick street in Oskaloosa, Iowa. The overlay used a Type II "
            "portland cement mix designed for a 28-day compressive strength of 4,500 psi. Pre-overlay "
            "condition surveys documented transverse cracking and rutting greater than 0.5 in along "
            "the wheel paths. Construction included a mill-and-fill of distressed asphalt patches "
            "before the overlay was placed. Performance was monitored via Falling Weight Deflectometer "
            "(FWD) testing and International Roughness Index (IRI) measurements. After 5 years of "
            "service under 1.2 million ESALs, IRI had increased from 95 in/mi to 142 in/mi. A Life "
            "Cycle Cost Analysis (LCCA) following the procedure in Chapter 3 of Publication 242 "
            "indicated whitetopping was the most cost-effective option over a 20-year horizon. "
            "Construction specifications are detailed in Publication 408, Section 501."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="treatment",
                extraction_text="ultrathin whitetopping overlay",
                attributes={"description": "rehabilitation treatment evaluated for use on a deteriorated brick street"},
            ),
            lx.data.Extraction(
                extraction_class="material",
                extraction_text="Type II portland cement mix",
                attributes={"description": "concrete mix used in the whitetopping overlay"},
            ),
            lx.data.Extraction(
                extraction_class="specification",
                extraction_text="28-day compressive strength of 4,500 psi",
                attributes={"description": "design strength specification for the overlay concrete mix"},
            ),
            lx.data.Extraction(
                extraction_class="distress",
                extraction_text="transverse cracking",
                attributes={"description": "pre-overlay distress recorded during condition surveys"},
            ),
            lx.data.Extraction(
                extraction_class="distress",
                extraction_text="rutting greater than 0.5 in along the wheel paths",
                attributes={"description": "pre-overlay rutting severity along wheel paths"},
            ),
            lx.data.Extraction(
                extraction_class="treatment",
                extraction_text="mill-and-fill",
                attributes={"description": "preparatory treatment applied to distressed asphalt patches before placing the overlay"},
            ),
            lx.data.Extraction(
                extraction_class="test_method",
                extraction_text="Falling Weight Deflectometer (FWD) testing",
                attributes={"description": "structural performance monitoring method used during the evaluation"},
            ),
            lx.data.Extraction(
                extraction_class="test_method",
                extraction_text="International Roughness Index (IRI) measurements",
                attributes={"description": "surface roughness monitoring method used during the evaluation"},
            ),
            lx.data.Extraction(
                extraction_class="metric",
                extraction_text="1.2 million ESALs",
                attributes={"description": "cumulative traffic loading over the 5-year monitoring period"},
            ),
            lx.data.Extraction(
                extraction_class="metric",
                extraction_text="IRI had increased from 95 in/mi to 142 in/mi",
                attributes={"description": "observed surface roughness degradation over 5 years of service"},
            ),
            lx.data.Extraction(
                extraction_class="process",
                extraction_text="Life Cycle Cost Analysis (LCCA)",
                attributes={"description": "decision framework used to compare treatment alternatives over a 20-year horizon"},
            ),
            lx.data.Extraction(
                extraction_class="reference",
                extraction_text="Chapter 3 of Publication 242",
                attributes={
                    "description": "LCCA procedure reference",
                    "title": "Publication 242, Pavement Design Manual, Chapter 3",
                    "source": "Publication 242, Pavement Design Manual",
                    "context": "Chapter 3 specifies the LCCA procedure followed in this study to compare treatment alternatives",
                },
            ),
            lx.data.Extraction(
                extraction_class="reference",
                extraction_text="Publication 408, Section 501",
                attributes={
                    "description": "construction specifications reference for the whitetopping overlay",
                    "title": "Publication 408, Specifications, Section 501",
                    "source": "Publication 408, Specifications",
                    "context": "Section 501 of Publication 408 provides the construction specifications for whitetopping referenced by this study",
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
    extraction_passes: int = 3
    max_char_buffer: int = 10000
    prompt_name: str = "nemo_logic-artifacts"
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
            entry: dict = {
                "artifact_id": f"{doc_id}_chunk_{chunk_id}_art_{ext_idx}",
                "text": ext.extraction_text,
                "char_interval": char_iv,
                "attributes": dict(ext.attributes or {}),
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
