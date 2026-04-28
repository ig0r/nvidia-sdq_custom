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


SPAN_LEVEL_CLASSES: list[str] = [
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

CHUNK_LEVEL_CLASSES: list[str] = [
    "chunk_summary",
    "chunk_topic",
    "pavement_engineering_term",
]

EXTRACTION_CLASSES: list[str] = SPAN_LEVEL_CLASSES + CHUNK_LEVEL_CLASSES


SPAN_LEVEL_EXAMPLES: list[lx.data.ExampleData] = [
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
    # Example 2 — manual/specification style (RCA mix design).
    # Demonstrates type-specific attribute keys for: definition, method, formula,
    # parameter, threshold, exception, constraint, procedure, assumption, rationale.
    lx.data.ExampleData(
        text=(
            "Recycled concrete aggregate, abbreviated RCA, is aggregate produced by "
            "crushing reclaimed hydraulic-cement concrete. The volumetric method "
            "displaces a known volume of water from a calibrated bowl to measure the "
            "entrained air content of fresh RCA concrete. The water-to-cementitious-"
            "materials ratio is computed as w/cm = m_w / m_cm, where m_w is the mass "
            "of mixing water and m_cm is the mass of all cementitious materials. The "
            "coarse RCA replacement level governs both fresh and hardened mixture "
            "properties and is expressed as a percentage on a volume basis. Coarse "
            "RCA replacement greater than 50 percent triggers a strength verification "
            "test prior to placement. RCA concrete is permitted in this specification "
            "except for projects where the source concrete is known to be affected by "
            "alkali-silica reaction. Use of RCA is limited to mixtures with a w/cm at "
            "or below 0.45. To proportion a trial RCA concrete mixture, the designer "
            "shall first batch a one-cubic-yard trial; then measure slump, air content, "
            "and unit weight; finally adjust the paste content and retest if any "
            "property falls outside the target range. For mechanistic-empirical rigid "
            "pavement design, the analysis assumes a representative 28-day flexural "
            "strength of 600 psi unless project-specific testing supports a different "
            "value, because trial batching is generally not available before the design "
            "phase."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="definition",
                extraction_text=(
                    "Recycled concrete aggregate, abbreviated RCA, is aggregate produced by "
                    "crushing reclaimed hydraulic-cement concrete"
                ),
                attributes={
                    "term": "Recycled concrete aggregate (RCA)",
                    "definition": "aggregate produced by crushing reclaimed hydraulic-cement concrete",
                    "description": "definition of recycled concrete aggregate (RCA)",
                },
            ),
            lx.data.Extraction(
                extraction_class="method",
                extraction_text=(
                    "The volumetric method displaces a known volume of water from a "
                    "calibrated bowl to measure the entrained air content of fresh RCA concrete"
                ),
                attributes={
                    "method_name": "volumetric method",
                    "purpose": "measure entrained air content of fresh RCA concrete",
                    "context": "testing",
                    "input": "known volume of water in a calibrated bowl",
                    "output": "entrained air content",
                    "description": "test method for measuring air content via water displacement",
                },
            ),
            lx.data.Extraction(
                extraction_class="formula",
                extraction_text="w/cm = m_w / m_cm",
                attributes={
                    "expression": "w/cm = m_w / m_cm",
                    "computes": "water-to-cementitious-materials ratio",
                    "variables": "m_w (mass of mixing water), m_cm (mass of cementitious materials)",
                    "description": "formula for the water-to-cementitious-materials ratio",
                },
            ),
            lx.data.Extraction(
                extraction_class="parameter",
                extraction_text=(
                    "The coarse RCA replacement level governs both fresh and hardened "
                    "mixture properties and is expressed as a percentage on a volume basis"
                ),
                attributes={
                    "name": "coarse RCA replacement level",
                    "unit": "percent (volume basis)",
                    "role": "design value",
                    "applies_to": "RCA concrete mix proportioning",
                    "description": "design parameter governing fresh and hardened RCA mixture properties",
                },
            ),
            lx.data.Extraction(
                extraction_class="threshold",
                extraction_text=(
                    "Coarse RCA replacement greater than 50 percent triggers a strength "
                    "verification test prior to placement"
                ),
                attributes={
                    "value": "50",
                    "unit": "percent",
                    "comparison": "greater than",
                    "applies_to": "coarse RCA replacement level",
                    "decision_effect": "triggers a strength verification test prior to placement",
                    "description": "boundary above which coarse RCA replacement requires strength verification",
                    "significance": "strength verification is required when coarse RCA replacement exceeds 50 percent",
                },
            ),
            lx.data.Extraction(
                extraction_class="exception",
                extraction_text=(
                    "RCA concrete is permitted in this specification except for projects "
                    "where the source concrete is known to be affected by alkali-silica reaction"
                ),
                attributes={
                    "normal_rule": "RCA concrete is permitted in this specification",
                    "exception_case": "projects where the source concrete is known to be affected by alkali-silica reaction",
                    "approval_required": "unspecified",
                    "source_cue": "except for",
                    "description": "exception barring RCA concrete when the source concrete is ASR-affected",
                },
            ),
            lx.data.Extraction(
                extraction_class="constraint",
                extraction_text="Use of RCA is limited to mixtures with a w/cm at or below 0.45",
                attributes={
                    "limitation": "w/cm at or below 0.45",
                    "applies_to": "mixtures containing RCA",
                    "scope": "all RCA concrete mixtures under this specification",
                    "source_cue": "limited to",
                    "description": "upper-bound constraint on the water-to-cementitious-materials ratio for RCA mixtures",
                },
            ),
            lx.data.Extraction(
                extraction_class="procedure",
                extraction_text=(
                    "To proportion a trial RCA concrete mixture, the designer shall first "
                    "batch a one-cubic-yard trial; then measure slump, air content, and unit "
                    "weight; finally adjust the paste content and retest if any property "
                    "falls outside the target range"
                ),
                attributes={
                    "procedure_name": "proportioning a trial RCA concrete mixture",
                    "step": "batch one-cubic-yard trial; measure slump, air content, and unit weight; adjust paste content and retest if any property is out of range",
                    "sequence_indicator": "first / then / finally",
                    "input": "initial mix proportions",
                    "output": "adjusted trial mix proportions",
                    "actor": "designer",
                    "description": "stepwise procedure for proportioning a trial RCA concrete mixture",
                },
            ),
            lx.data.Extraction(
                extraction_class="assumption",
                extraction_text=(
                    "the analysis assumes a representative 28-day flexural strength of 600 "
                    "psi unless project-specific testing supports a different value"
                ),
                attributes={
                    "assumption": "representative 28-day flexural strength of 600 psi",
                    "assumed_value": "600 psi",
                    "scope": "mechanistic-empirical rigid pavement design",
                    "basis": "default value used unless project-specific testing supports a different value",
                    "description": "default 28-day flexural strength assumption for ME pavement design with RCA concrete",
                },
            ),
            lx.data.Extraction(
                extraction_class="rationale",
                extraction_text=(
                    "because trial batching is generally not available before the design phase"
                ),
                attributes={
                    "explains": "use of an assumed flexural strength rather than a measured value",
                    "reason": "trial batching is generally not available before the design phase",
                    "source_cue": "because",
                    "description": "rationale for using a default flexural strength rather than a measured value",
                },
            ),
        ],
    ),
    # Example 3 — report/research style (RCA strength + brick-street rehabilitation).
    # Demonstrates type-specific attribute keys for: evidence, finding, issue,
    # rationale, decision, recommendation, risk.
    lx.data.ExampleData(
        text=(
            "A statistical study using data from more than 100 peer-reviewed studies "
            "indicated that RCA concrete typically exhibits a 10 to 15 percent reduction "
            "in compressive strength compared with companion conventional concrete "
            "mixtures. Transverse cracking is the dominant distress observed in the "
            "existing pavement section. This cracking results from the brittleness of "
            "the brick base combined with the flexibility of the asphalt overlay. The "
            "investigation team selected the ultrathin whitetopping option over the "
            "asphaltic concrete overlay alternative because the whitetopping demonstrated "
            "better long-term performance in similar climates. Field deflection testing "
            "using the falling weight deflectometer recorded deflections under 8 mils "
            "across all test locations on the Oskaloosa project. Practitioners should "
            "monitor stockpile moisture content closely, since sudden drops in moisture "
            "have been linked to rapid workability loss during paving. Use of high-"
            "replacement fine RCA in interstate shoulders introduces an elevated risk of "
            "premature surface scaling under freeze-thaw exposure. Engineers should "
            "consider blending RCA with virgin aggregate when the source concrete is "
            "variable, because blending reduces the proportion of adhered mortar and "
            "stabilizes the strength of the resulting mixture."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="evidence",
                extraction_text="A statistical study using data from more than 100 peer-reviewed studies",
                attributes={
                    "evidence_type": "data",
                    "evidence_value": "more than 100 peer-reviewed studies",
                    "supports": "finding on RCA compressive strength reduction",
                    "description": "statistical evidence base for findings on RCA concrete strength",
                },
            ),
            lx.data.Extraction(
                extraction_class="finding",
                extraction_text=(
                    "RCA concrete typically exhibits a 10 to 15 percent reduction in "
                    "compressive strength compared with companion conventional concrete mixtures"
                ),
                attributes={
                    "finding_subject": "compressive strength of RCA concrete",
                    "finding_value": "10 to 15 percent reduction compared with companion conventional concrete mixtures",
                    "basis": "statistical study of more than 100 peer-reviewed studies",
                    "severity": "moderate",
                    "description": "observed reduction in RCA concrete compressive strength relative to conventional mixtures",
                },
            ),
            lx.data.Extraction(
                extraction_class="issue",
                extraction_text=(
                    "Transverse cracking is the dominant distress observed in the existing pavement section"
                ),
                attributes={
                    "problem": "transverse cracking",
                    "problem_type": "design",
                    "affected_item": "existing pavement section",
                    "severity": "high",
                    "description": "transverse cracking identified as the dominant distress in the existing pavement",
                },
            ),
            lx.data.Extraction(
                extraction_class="rationale",
                extraction_text=(
                    "This cracking results from the brittleness of the brick base combined "
                    "with the flexibility of the asphalt overlay"
                ),
                attributes={
                    "explains": "presence of transverse cracking in the existing pavement",
                    "reason": "brittleness of the brick base combined with the flexibility of the asphalt overlay",
                    "source_cue": "results from",
                    "description": "explanation linking transverse cracking to the brittle/flexible base-overlay mismatch",
                },
            ),
            lx.data.Extraction(
                extraction_class="decision",
                extraction_text=(
                    "The investigation team selected the ultrathin whitetopping option over "
                    "the asphaltic concrete overlay alternative"
                ),
                attributes={
                    "decision_subject": "rehabilitation alternative for the brick street",
                    "selected_option": "ultrathin whitetopping",
                    "rejected_option": "asphaltic concrete overlay",
                    "authority": "investigation team",
                    "basis": "whitetopping demonstrated better long-term performance in similar climates",
                    "source_cue": "selected",
                    "description": "selection of ultrathin whitetopping over asphaltic concrete overlay",
                },
            ),
            lx.data.Extraction(
                extraction_class="finding",
                extraction_text=(
                    "Field deflection testing using the falling weight deflectometer recorded "
                    "deflections under 8 mils across all test locations on the Oskaloosa project"
                ),
                attributes={
                    "finding_subject": "pavement deflections at FWD test locations",
                    "finding_value": "deflections under 8 mils across all test locations on the Oskaloosa project",
                    "basis": "field falling weight deflectometer testing",
                    "severity": "unspecified",
                    "description": "in-field deflection finding at the Oskaloosa project test locations",
                },
            ),
            lx.data.Extraction(
                extraction_class="recommendation",
                extraction_text="Practitioners should monitor stockpile moisture content closely",
                attributes={
                    "recommended_action": "monitor stockpile moisture content closely",
                    "target": "RCA stockpile",
                    "strength": "should",
                    "rationale": "sudden drops in moisture have been linked to rapid workability loss during paving",
                    "source_cue": "should",
                    "description": "recommendation to closely monitor RCA stockpile moisture content",
                    "significance": "rapid workability loss during paving has been linked to sudden drops in stockpile moisture",
                },
            ),
            lx.data.Extraction(
                extraction_class="risk",
                extraction_text=(
                    "Use of high-replacement fine RCA in interstate shoulders introduces an "
                    "elevated risk of premature surface scaling under freeze-thaw exposure"
                ),
                attributes={
                    "risk_event": "premature surface scaling",
                    "consequence": "reduced surface durability",
                    "affected_item": "interstate shoulders using high-replacement fine RCA",
                    "likelihood": "elevated",
                    "severity": "moderate",
                    "description": "elevated scaling risk for interstate shoulders using high-replacement fine RCA",
                    "significance": "scaling can occur under freeze-thaw exposure",
                },
            ),
            lx.data.Extraction(
                extraction_class="recommendation",
                extraction_text=(
                    "Engineers should consider blending RCA with virgin aggregate when the "
                    "source concrete is variable"
                ),
                attributes={
                    "recommended_action": "blending RCA with virgin aggregate",
                    "target": "RCA concrete mixture",
                    "condition": "source concrete is variable",
                    "strength": "should consider",
                    "rationale": "blending reduces the proportion of adhered mortar and stabilizes mixture strength",
                    "source_cue": "should consider",
                    "description": "recommendation to blend RCA with virgin aggregate when source variability is present",
                    "significance": "blending reduces adhered mortar and stabilizes the strength of the resulting mixture",
                },
            ),
        ],
    ),
]


CHUNK_LEVEL_EXAMPLES: list[lx.data.ExampleData] = [
    # Example A — manual/spec style. Same text as SPAN_LEVEL_EXAMPLES[1] (RCA mix design),
    # demonstrating chunk_summary span = last complete sentence of the chunk.
    lx.data.ExampleData(
        text=(
            "Recycled concrete aggregate, abbreviated RCA, is aggregate produced by "
            "crushing reclaimed hydraulic-cement concrete. The volumetric method "
            "displaces a known volume of water from a calibrated bowl to measure the "
            "entrained air content of fresh RCA concrete. The water-to-cementitious-"
            "materials ratio is computed as w/cm = m_w / m_cm, where m_w is the mass "
            "of mixing water and m_cm is the mass of all cementitious materials. The "
            "coarse RCA replacement level governs both fresh and hardened mixture "
            "properties and is expressed as a percentage on a volume basis. Coarse "
            "RCA replacement greater than 50 percent triggers a strength verification "
            "test prior to placement. RCA concrete is permitted in this specification "
            "except for projects where the source concrete is known to be affected by "
            "alkali-silica reaction. Use of RCA is limited to mixtures with a w/cm at "
            "or below 0.45. To proportion a trial RCA concrete mixture, the designer "
            "shall first batch a one-cubic-yard trial; then measure slump, air content, "
            "and unit weight; finally adjust the paste content and retest if any "
            "property falls outside the target range. For mechanistic-empirical rigid "
            "pavement design, the analysis assumes a representative 28-day flexural "
            "strength of 600 psi unless project-specific testing supports a different "
            "value, because trial batching is generally not available before the design "
            "phase."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="chunk_summary",
                extraction_text=(
                    "For mechanistic-empirical rigid pavement design, the analysis "
                    "assumes a representative 28-day flexural strength of 600 psi "
                    "unless project-specific testing supports a different value, "
                    "because trial batching is generally not available before the "
                    "design phase."
                ),
                attributes={
                    "summary": (
                        "The chunk specifies how to proportion RCA concrete: defines "
                        "RCA, names the volumetric air-content test method, gives the "
                        "w/cm formula, sets replacement-level limits and an exception "
                        "for ASR-affected source concrete, describes the trial-mix "
                        "procedure, and states the default 28-day flexural strength "
                        "assumption used for ME pavement design."
                    ),
                    "document_function": "definition and material guidance",
                    "scope": "RCA concrete mix design and rigid pavement design with RCA",
                },
            ),
            lx.data.Extraction(
                extraction_class="chunk_topic",
                extraction_text="trial RCA concrete mixture",
                attributes={
                    "topic": "RCA concrete mix design",
                    "topic_role": "primary",
                },
            ),
            lx.data.Extraction(
                extraction_class="chunk_topic",
                extraction_text="mechanistic-empirical rigid pavement design",
                attributes={
                    "topic": "rigid pavement design",
                    "topic_role": "secondary",
                },
            ),
            lx.data.Extraction(
                extraction_class="chunk_topic",
                extraction_text="alkali-silica reaction",
                attributes={
                    "topic": "materials durability",
                    "topic_role": "secondary",
                },
            ),
            lx.data.Extraction(
                extraction_class="pavement_engineering_term",
                extraction_text="RCA",
                attributes={
                    "term": "RCA",
                    "normalized_term": "recycled concrete aggregate",
                    "term_category": "material",
                },
            ),
            lx.data.Extraction(
                extraction_class="pavement_engineering_term",
                extraction_text="w/cm",
                attributes={
                    "term": "w/cm",
                    "normalized_term": "water-to-cementitious-materials ratio",
                    "term_category": "design_parameter",
                },
            ),
            lx.data.Extraction(
                extraction_class="pavement_engineering_term",
                extraction_text="28-day flexural strength",
                attributes={
                    "term": "28-day flexural strength",
                    "normalized_term": "28-day flexural strength",
                    "term_category": "design_parameter",
                },
            ),
        ],
    ),
    # Example B — report style. Same text as SPAN_LEVEL_EXAMPLES[2] (RCA + brick-street findings).
    lx.data.ExampleData(
        text=(
            "A statistical study using data from more than 100 peer-reviewed studies "
            "indicated that RCA concrete typically exhibits a 10 to 15 percent reduction "
            "in compressive strength compared with companion conventional concrete "
            "mixtures. Transverse cracking is the dominant distress observed in the "
            "existing pavement section. This cracking results from the brittleness of "
            "the brick base combined with the flexibility of the asphalt overlay. The "
            "investigation team selected the ultrathin whitetopping option over the "
            "asphaltic concrete overlay alternative because the whitetopping demonstrated "
            "better long-term performance in similar climates. Field deflection testing "
            "using the falling weight deflectometer recorded deflections under 8 mils "
            "across all test locations on the Oskaloosa project. Practitioners should "
            "monitor stockpile moisture content closely, since sudden drops in moisture "
            "have been linked to rapid workability loss during paving. Use of high-"
            "replacement fine RCA in interstate shoulders introduces an elevated risk of "
            "premature surface scaling under freeze-thaw exposure. Engineers should "
            "consider blending RCA with virgin aggregate when the source concrete is "
            "variable, because blending reduces the proportion of adhered mortar and "
            "stabilizes the strength of the resulting mixture."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="chunk_summary",
                extraction_text=(
                    "Engineers should consider blending RCA with virgin aggregate when "
                    "the source concrete is variable, because blending reduces the "
                    "proportion of adhered mortar and stabilizes the strength of the "
                    "resulting mixture."
                ),
                attributes={
                    "summary": (
                        "The chunk reports an observed reduction in RCA-concrete "
                        "compressive strength relative to conventional mixtures, "
                        "identifies transverse cracking and its cause in an existing "
                        "brick-based pavement, documents the team's selection of "
                        "ultrathin whitetopping over an asphaltic concrete overlay, "
                        "and recommends practices for stockpile moisture monitoring "
                        "and RCA blending where source concrete is variable."
                    ),
                    "document_function": "finding, decision, and recommendation",
                    "scope": "RCA concrete pavement performance and brick-street rehabilitation",
                },
            ),
            lx.data.Extraction(
                extraction_class="chunk_topic",
                extraction_text="whitetopping option",
                attributes={
                    "topic": "pavement rehabilitation",
                    "topic_role": "primary",
                },
            ),
            lx.data.Extraction(
                extraction_class="chunk_topic",
                extraction_text="compressive strength",
                attributes={
                    "topic": "materials performance",
                    "topic_role": "secondary",
                },
            ),
            lx.data.Extraction(
                extraction_class="chunk_topic",
                extraction_text="blending RCA with virgin aggregate",
                attributes={
                    "topic": "RCA concrete mix design",
                    "topic_role": "secondary",
                },
            ),
            lx.data.Extraction(
                extraction_class="pavement_engineering_term",
                extraction_text="RCA",
                attributes={
                    "term": "RCA",
                    "normalized_term": "recycled concrete aggregate",
                    "term_category": "material",
                },
            ),
            lx.data.Extraction(
                extraction_class="pavement_engineering_term",
                extraction_text="falling weight deflectometer",
                attributes={
                    "term": "falling weight deflectometer",
                    "normalized_term": "falling weight deflectometer",
                    "term_category": "test_method",
                },
            ),
            lx.data.Extraction(
                extraction_class="pavement_engineering_term",
                extraction_text="ultrathin whitetopping",
                attributes={
                    "term": "ultrathin whitetopping",
                    "normalized_term": "ultrathin whitetopping",
                    "term_category": "construction",
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
    extraction_passes: int = 2                # span-level recall passes
    chunk_extraction_passes: int = 1          # chunk-level passes (deterministic)
    max_char_buffer: int = 10000
    prompt_name_span: str = "nemo_logic-artifacts-03-span"
    prompt_name_chunk: str = "nemo_logic-artifacts-03-chunk"
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
        span_path: Path = Path(cfg.prompt_lib) / f"{cfg.prompt_name_span}.txt"
        chunk_path: Path = Path(cfg.prompt_lib) / f"{cfg.prompt_name_chunk}.txt"
        missing: list[str] = [str(p) for p in (span_path, chunk_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Prompt(s) not found: {missing}")
        self.prompt_span: str = span_path.read_text(encoding="utf-8")
        self.prompt_chunk: str = chunk_path.read_text(encoding="utf-8")

    def _char_iv(self, ext) -> dict | None:
        ci = ext.char_interval
        return (
            {"start_pos": ci.start_pos, "end_pos": ci.end_pos}
            if ci is not None
            else None
        )

    def extract(self, text: str, doc_id: str, chunk_id: int) -> dict[str, list[dict]]:
        # Span-level call: 21 classes, v2 prompt body, recall passes.
        span_result = lx.extract(
            text_or_documents=text,
            prompt_description=self.prompt_span,
            examples=SPAN_LEVEL_EXAMPLES,
            model_id=self.cfg.model,
            api_key=self.api_key,
            temperature=self.cfg.temperature,
            extraction_passes=self.cfg.extraction_passes,
            max_char_buffer=self.cfg.max_char_buffer,
            show_progress=False,
        )
        # Chunk-level call: 3 classes, dedicated prompt, deterministic passes.
        chunk_result = lx.extract(
            text_or_documents=text,
            prompt_description=self.prompt_chunk,
            examples=CHUNK_LEVEL_EXAMPLES,
            model_id=self.cfg.model,
            api_key=self.api_key,
            temperature=self.cfg.temperature,
            extraction_passes=self.cfg.chunk_extraction_passes,
            max_char_buffer=self.cfg.max_char_buffer,
            show_progress=False,
        )
        bucketed: dict[str, list[dict]] = {}
        ext_idx: int = 0
        # Span-level: gate against SPAN_LEVEL_CLASSES; promote description/significance.
        for ext in (span_result.extractions or []):
            cls: str = ext.extraction_class
            if cls not in SPAN_LEVEL_CLASSES:
                logger.log(
                    "NLP",
                    f"{doc_id} chunk {chunk_id}: span call dropped class {cls!r}",
                )
                continue
            attrs: dict = dict(ext.attributes or {})
            description: str = attrs.pop("description", "") or ""
            significance: str | None = attrs.pop("significance", None) or None
            entry: dict = {
                "artifact_id": f"{doc_id}_chunk_{chunk_id}_art_{ext_idx}",
                "text": ext.extraction_text,
                "description": description,
                "significance": significance,
                "char_interval": self._char_iv(ext),
                "attributes": attrs,
            }
            bucketed.setdefault(cls, []).append(entry)
            ext_idx += 1
        # Chunk-level: gate against CHUNK_LEVEL_CLASSES; bypass description/significance promotion.
        for ext in (chunk_result.extractions or []):
            cls = ext.extraction_class
            if cls not in CHUNK_LEVEL_CLASSES:
                logger.log(
                    "NLP",
                    f"{doc_id} chunk {chunk_id}: chunk call dropped class {cls!r}",
                )
                continue
            entry = {
                "artifact_id": f"{doc_id}_chunk_{chunk_id}_art_{ext_idx}",
                "text": ext.extraction_text,
                "char_interval": self._char_iv(ext),
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
