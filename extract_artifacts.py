#!/usr/bin/env python3
"""Standalone CLI: extract pavement-engineering artifacts from logical contexts.

Reads {output_dir}/doc-chunks_{size}_random_logical/*-logic-ctx.json (produced by
`_nemo.py --sdg-logical`). Per context it runs two extraction calls and writes
both into *-logic-artifacts.json:
- Span-level (langextract → gpt-4o-mini): 21-class normative taxonomy with verbatim spans.
- Chunk-level (OpenAI Structured Outputs → gpt-4o-mini): a single Pydantic-validated
  ChunkSignals object (summary + 1-5 topics + 0+ pavement_engineering_terms).

The context's text is the join of `chunks[*].text` with `"\\n\\n"` — same formula
as `generate-qa.py::build_context_text`. Each output record is keyed by `u_ctx_id`.
For 1:1 contexts (current `_build_logical_contexts` behavior) this is byte-equivalent
to a per-chunk run modulo model nondeterminism.

Requires:
- [chunking].method == 'random_logical' in the cfg TOML.
- *-logic-ctx.json must exist per doc (run `_nemo.py --sdg-logical` first).
- OPENAI_API_KEY in env or .env.
- pip install langextract openai pydantic loguru python-dotenv

Single-file standalone: no imports from the project's aisa/ package.
"""
import argparse
import json
import os
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from loguru import logger
import langextract as lx
import langextract.providers.ollama as _lx_ollama
from ollama import Client as OllamaClient
from openai import OpenAI
from pydantic import BaseModel, Field


load_dotenv()

# Custom log levels matching aisa/utils/log.py so output style stays consistent
# with the rest of the project. logger.level() raises if the level already exists,
# which can happen if this module is imported alongside the aisa logger.
for _name, _no, _icon in (("CHUNK", 14, "📦"), ("NLP", 10, "🧠")):
    try:
        logger.level(_name, no=_no, icon=_icon)
    except (TypeError, ValueError):
        pass


# ---------------------------------------------------------------------------
# Model routing (gpt-* / gemini-* / ollama).
# Keep byte-identical to generate-qa.py:171-185 (SRS FR-7.2); if one changes,
# change both — the two copies must not drift.
# ---------------------------------------------------------------------------

def is_ollama_model(model: str) -> bool:
    # Ollama identifiers use `name:tag` (e.g. "qwen3:4b", "gpt-oss:120b",
    # "hf.co/owner/repo:tag"). OpenAI and Gemini model IDs never use a colon,
    # so this is the unambiguous signal and it dominates prefix-based checks.
    if ":" in model:
        return True
    return not (model.startswith("gpt") or model.startswith("gemini"))


def is_gpt(model: str) -> bool:
    return model.startswith("gpt") and not is_ollama_model(model)


def is_gemini(model: str) -> bool:
    return model.startswith("gemini") and not is_ollama_model(model)


def _resolve_ollama_host(cli_host: str | None, cfg_host: str | None) -> str:
    """Ollama endpoint precedence: CLI (--host/--port) > config
    [artifact_extraction].ollama_host > $OLLAMA_HOST (cluster slurm exports it)
    > http://localhost:11434. Used by both the chunk client and _OllamaSpanLM."""
    h = (
        cli_host
        or cfg_host
        or os.environ.get("OLLAMA_HOST")
        or "http://localhost:11434"
    )
    if not h.startswith(("http://", "https://")):
        h = f"http://{h}"
    return h


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

EXTRACTION_CLASSES: list[str] = SPAN_LEVEL_CLASSES   # alias for back-compat


# Chunk-level Pydantic schema (used as response_format for OpenAI Structured Outputs).
DOCUMENT_FUNCTION = Literal[
    "requirement",
    "procedure",
    "design guidance",
    "calculation guidance",
    "definition",
    "approval workflow",
    "material guidance",
    "construction guidance",
    "testing guidance",
    "maintenance guidance",
    "finding",
    "recommendation",
    "rationale",
    "issue",
    "risk",
    "evidence",
    "example",
]
TOPIC_ROLE = Literal["primary", "secondary"]
TERM_CATEGORY = Literal[
    "traffic",
    "pavement_type",
    "design_parameter",
    "material",
    "layer",
    "method",
    "test_method",
    "distress",
    "construction",
    "maintenance",
    "organization",
    "form",
    "software",
    "other",
]


class ChunkSummary(BaseModel):
    """Summary of the whole chunk."""
    summary: str = Field(
        description=(
            "One or two sentence content statement, source-grounded and self-contained. "
            "Do not refer to the chunk, passage, or text."
        )
    )
    document_functions: list[DOCUMENT_FUNCTION] = Field(
        description="One or more roles the chunk plays in the document. Pick all that apply."
    )
    scope: str | None = Field(
        description="Where, when, or to what the chunk applies, if stated. Null otherwise."
    )


class ChunkTopic(BaseModel):
    """A normalized topic label for a major subject of the chunk."""
    topic: str = Field(description="Concise normalized topic label, not a full sentence.")
    role: TOPIC_ROLE = Field(
        description="primary for the main topic; secondary for other important topics."
    )


class PavementTerm(BaseModel):
    """A pavement engineering term explicitly present in the chunk."""
    term: str = Field(description="The term as it appears in the source (verbatim).")
    normalized_term: str | None = Field(
        description="Canonical version of the term, when useful. Null if already canonical."
    )
    category: TERM_CATEGORY = Field(description="Category of the term.")


class ChunkSignals(BaseModel):
    """Chunk-level signals: a single summary, 1-5 topics, 0+ pavement engineering terms."""
    summary: ChunkSummary
    topics: list[ChunkTopic] = Field(
        description="One to five topics. Exactly one has role='primary'; the rest are 'secondary'."
    )
    terms: list[PavementTerm] = Field(
        description="Zero or more pavement engineering terms explicitly present in the chunk."
    )


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


@dataclass
class LXConfig:
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    temperature: float = 0.0
    extraction_passes: int = 2                              # span-level recall passes
    chunk_concurrency: int = 8                              # per-doc thread pool size (tier-2 default)
    max_char_buffer: int = 10000
    prompt_name: str = "nemo_logic-artifacts-04-span"        # span-level (langextract)
    chunk_prompt_name: str = "nemo_logic-artifacts-04-chunk" # chunk-level (OpenAI Structured Outputs)
    prompt_lib: str = "./prompts"
    # Ollama path (selected when is_ollama_model(model), e.g. "gpt-oss:20b/120b"). SRS FR-2.2.
    # ollama_host resolved in main(): CLI --host/--port > toml > $OLLAMA_HOST > localhost.
    ollama_host: str | None = None
    ollama_num_ctx: int = 16384
    ollama_timeout: int = 600
    ollama_retries: int = 3


class ChunkLevelExtractor:
    """Direct OpenAI Structured Outputs call producing a ChunkSignals object per chunk.

    Quantity rules are soft-validated post-receipt: cap topics at 5; log if 0 or > 5.
    """

    def __init__(self, cfg: LXConfig):
        self.cfg: LXConfig = cfg
        self.is_ollama: bool = is_ollama_model(cfg.model)
        prompt_path: Path = Path(cfg.prompt_lib) / f"{cfg.chunk_prompt_name}.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Chunk prompt not found: {prompt_path}")
        self.system_prompt: str = prompt_path.read_text(encoding="utf-8")
        if self.is_ollama:
            self.client = OllamaClient(host=cfg.ollama_host)
        else:
            api_key: str | None = cfg.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY not set (env or [artifact_extraction].api_key in cfg)"
                )
            # max_retries=5 (default 2) absorbs burst 429s under v5 thread concurrency.
            self.client = OpenAI(api_key=api_key, max_retries=5)

    def extract(self, text: str, doc_id: str, chunk_id: int) -> ChunkSignals:
        if self.is_ollama:
            signals: ChunkSignals = self._extract_ollama(text, doc_id, chunk_id)
        else:
            completion = self.client.beta.chat.completions.parse(
                model=self.cfg.model,
                temperature=self.cfg.temperature,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text},
                ],
                response_format=ChunkSignals,
            )
            signals = completion.choices[0].message.parsed
            if signals is None:
                raise RuntimeError(
                    f"OpenAI returned no parsed ChunkSignals "
                    f"(refusal: {completion.choices[0].message.refusal!r})"
                )
        # Soft validation of topics quantity rule (1-5 expected).
        n_topics: int = len(signals.topics)
        if n_topics == 0:
            logger.log(
                "NLP",
                f"{doc_id} chunk {chunk_id}: 0 chunk topics emitted (expected 1-5)",
            )
        elif n_topics > 5:
            logger.log(
                "NLP",
                f"{doc_id} chunk {chunk_id}: {n_topics} chunk topics emitted; capping to first 5",
            )
            signals.topics = signals.topics[:5]
        return signals

    def _extract_ollama(self, text: str, doc_id: str, chunk_id: int) -> ChunkSignals:
        """Ollama chunk-level via the native client: grammar-constrained JSON
        (format=<schema>) + think=False (gpt-oss is a reasoning model). Retries
        on validation/transport errors; raises on exhaustion (preserves the
        orchestrator's raise-on-failure contract)."""
        schema = ChunkSignals.model_json_schema()
        last_exc: Exception | None = None
        for attempt in range(1, self.cfg.ollama_retries + 1):
            try:
                resp = self.client.chat(
                    model=self.cfg.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": text},
                    ],
                    options={
                        "temperature": self.cfg.temperature,
                        "num_ctx": self.cfg.ollama_num_ctx,
                    },
                    format=schema,
                    think=False,
                )
                content = (
                    resp["message"]["content"]
                    if isinstance(resp, dict)
                    else resp.message.content
                )
                return ChunkSignals.model_validate_json(content)
            except Exception as exc:  # pydantic ValidationError + ollama errors
                last_exc = exc
                logger.log(
                    "NLP",
                    f"{doc_id} chunk {chunk_id}: ollama attempt "
                    f"{attempt}/{self.cfg.ollama_retries} failed: {exc}",
                )
        raise RuntimeError(
            f"Ollama ChunkSignals failed after {self.cfg.ollama_retries} "
            f"attempts: {last_exc}"
        )


class _OllamaSpanLM(_lx_ollama.OllamaLanguageModel):
    """langextract Ollama model with the hardcoded response `format` removed.

    Stock langextract always sends ``format: json|yaml`` on ``/api/generate`` and
    never suppresses thinking. gpt-oss (Harmony reasoning — 20b *and* 120b) then
    emits chain-of-thought instead of JSON under that grammar, so 0 spans parse.
    With the forced grammar removed, the prompt + few-shot examples elicit clean
    JSON that langextract's resolver parses normally (ablation-verified: 6–7
    valid spans vs 0). Passed via ``lx.extract(model=...)`` so langextract uses
    it as-is and does not re-apply schema constraints. OpenAI span path unchanged.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Stop _ollama_query() re-deriving a forced format from self.format_type.
        self.format_type = None

    def infer(self, batch_prompts, **kwargs):
        combined = self.merge_kwargs(kwargs)
        for prompt in batch_prompts:
            resp = self._ollama_query(
                prompt=prompt,
                model=self._model,
                structured_output_format=None,  # → no payload['format']
                model_url=self._model_url,
                **combined,
            )
            yield [
                _lx_ollama.core_types.ScoredOutput(
                    score=1.0, output=resp["response"]
                )
            ]


class PavementExtractor:
    """Per-chunk orchestrator. Span-level via langextract; chunk-level via ChunkLevelExtractor.

    Each call is independently failure-isolated: span exceptions populate errors['span'] and
    leave extractions={}; chunk exceptions populate errors['chunk'] and leave chunk_signals=None.
    """

    def __init__(self, cfg: LXConfig):
        self.cfg: LXConfig = cfg
        self.is_ollama: bool = is_ollama_model(cfg.model)
        if self.is_ollama:
            self.api_key: str | None = None
        else:
            api_key: str | None = cfg.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY not set (env or [artifact_extraction].api_key in cfg)"
                )
            self.api_key = api_key
        prompt_path: Path = Path(cfg.prompt_lib) / f"{cfg.prompt_name}.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Span prompt not found: {prompt_path}")
        self.prompt_span: str = prompt_path.read_text(encoding="utf-8")
        self.chunk_extractor: ChunkLevelExtractor = ChunkLevelExtractor(cfg)

    def _char_iv(self, ext) -> dict | None:
        ci = ext.char_interval
        return (
            {"start_pos": ci.start_pos, "end_pos": ci.end_pos}
            if ci is not None
            else None
        )

    def _extract_spans(
        self, text: str, doc_id: str, chunk_id: int, u_ctx_id: str
    ) -> dict[str, list[dict]]:
        if self.is_ollama:
            # Our subclass strips langextract's forced `format` grammar (the
            # gpt-oss reasoning collision — SRS R2). Passed via model= so
            # langextract uses it as-is. num_ctx counteracts langextract's
            # 2048 default (SRS R1).
            span_lm = _OllamaSpanLM(
                model_id=self.cfg.model,
                base_url=self.cfg.ollama_host,
                temperature=self.cfg.temperature,
                num_ctx=self.cfg.ollama_num_ctx,
                timeout=self.cfg.ollama_timeout,
            )
            result = lx.extract(
                text_or_documents=text,
                prompt_description=self.prompt_span,
                examples=SPAN_LEVEL_EXAMPLES,
                model=span_lm,
                extraction_passes=self.cfg.extraction_passes,
                max_char_buffer=self.cfg.max_char_buffer,
                show_progress=False,
            )
        else:
            result = lx.extract(
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
        bucketed: dict[str, list[dict]] = {}
        ext_idx: int = 0
        for ext in (result.extractions or []):
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
                "u_artifact_id": f"{u_ctx_id}-art-{ext_idx}",
                "text": ext.extraction_text,
                "description": description,
                "significance": significance,
                "char_interval": self._char_iv(ext),
                "attributes": attrs,
            }
            bucketed.setdefault(cls, []).append(entry)
            ext_idx += 1
        return bucketed

    def extract(self, text: str, doc_id: str, chunk_id: int, u_ctx_id: str) -> dict:
        """Returns {extractions, chunk_signals, errors}.

        - extractions: dict[str, list[dict]]; span-level only, gated by SPAN_LEVEL_CLASSES.
        - chunk_signals: ChunkSignals.model_dump() or None on chunk-call failure.
        - errors: dict with keys 'span' and 'chunk'; each null on success or str(exc) on failure.

        Span and chunk calls run concurrently in a 2-worker thread pool (v5).
        Per-call failure isolation is preserved — each future's .result() is
        awaited under its own try/except.
        """
        extractions: dict[str, list[dict]] = {}
        chunk_signals: dict | None = None
        errors: dict[str, str | None] = {"span": None, "chunk": None}

        def _span_task() -> dict[str, list[dict]]:
            return self._extract_spans(text, doc_id, chunk_id, u_ctx_id)

        def _chunk_task() -> dict:
            signals: ChunkSignals = self.chunk_extractor.extract(text, doc_id, chunk_id)
            return signals.model_dump()

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"chunk{chunk_id}") as pool:
            f_span = pool.submit(_span_task)
            f_chunk = pool.submit(_chunk_task)
            try:
                extractions = f_span.result()
            except Exception as exc:
                logger.log(
                    "NLP",
                    f"{doc_id} chunk {chunk_id}: span extraction failed: {exc}",
                )
                errors["span"] = str(exc)
            try:
                chunk_signals = f_chunk.result()
            except Exception as exc:
                logger.log(
                    "NLP",
                    f"{doc_id} chunk {chunk_id}: chunk-signals extraction failed: {exc}",
                )
                errors["chunk"] = str(exc)

        return {
            "extractions": extractions,
            "chunk_signals": chunk_signals,
            "errors": errors,
        }


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def _resolve_input_dir(cfg: dict) -> Path:
    """Return the directory holding *-logic-ctx.json. Prefers [paths].input_dir
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


def main(cfg: dict, overwrite: bool = False, ollama_host: str | None = None) -> None:
    # If [chunking].method is present (cfg/nemo.toml shape), enforce mode 3.
    # If absent (extract_artifacts.toml shape), mode 3 is implied.
    method: str = cfg.get("chunking", {}).get("method", "random_logical")
    if method != "random_logical":
        raise ValueError(
            f"extract_artifacts requires [chunking].method == 'random_logical'; got {method!r}"
        )
    if "artifact_extraction" not in cfg and "langextract" in cfg:
        raise KeyError(
            "config has [langextract] but expects [artifact_extraction] (v3 → v4 rename). "
            "Update extract_artifacts.toml accordingly."
        )
    chunk_dir: Path = _resolve_input_dir(cfg)
    if not chunk_dir.exists():
        logger.log("NLP", f"Chunk dir not found: {chunk_dir}; nothing to do")
        return

    ctx_files: list[Path] = sorted(chunk_dir.glob("*-logic-ctx.json"))
    if not ctx_files:
        logger.log(
            "NLP",
            f"No *-logic-ctx.json under {chunk_dir}; nothing to do "
            f"(run `_nemo.py --sdg-logical` first to produce them)",
        )
        return

    lx_cfg: LXConfig = LXConfig(**cfg.get("artifact_extraction", {}))
    lx_cfg.ollama_host = _resolve_ollama_host(ollama_host, lx_cfg.ollama_host)
    extractor: PavementExtractor = PavementExtractor(lx_cfg)

    for ctx_path in ctx_files:
        ctx_doc: dict = _read_json(ctx_path)
        # The producer (`_nemo.py::_build_logical_contexts`) writes the canonical
        # doc_id into the JSON. Treat the JSON field as the authority; fall back
        # to filename derivation only if missing/empty. Soft-warn on mismatch
        # (usually a rename/relocation drift signal).
        filename_id: str = ctx_path.name.replace("-logic-ctx.json", "")
        json_id: str = ctx_doc.get("doc_id") or ""
        doc_id: str = json_id or filename_id
        if json_id and json_id != filename_id:
            logger.log(
                "NLP",
                f"{ctx_path}: doc_id mismatch — JSON says {json_id!r}, "
                f"filename derives {filename_id!r}; using JSON field",
            )
        out_path: Path = ctx_path.with_name(f"{doc_id}-logic-artifacts.json")

        if out_path.exists() and not overwrite:
            logger.log("CHUNK", f"{doc_id}: cache hit -> {out_path}")
            continue

        contexts: list[dict] = ctx_doc.get("contexts", [])

        # v6: iterate contexts (not chunks). Build per-context text via the
        # same formula as generate-qa.py::build_context_text. Chunks within a
        # context are joined with "\n\n" — for 1:1 contexts (current state) this
        # is identical to the chunk's own text. Order preserved by iterating the
        # futures list in submission order (matches v5 ordering invariant).
        artifacts: list[dict] = []
        with ThreadPoolExecutor(
            max_workers=lx_cfg.chunk_concurrency,
            thread_name_prefix=doc_id,
        ) as pool:
            futures: list[tuple] = []
            for ctx_entry in contexts:
                u_ctx_id: str = ctx_entry.get("u_ctx_id", "")
                chunks: list[dict] = ctx_entry.get("chunks", [])
                if not chunks:
                    logger.log(
                        "NLP",
                        f"{doc_id}: empty context {u_ctx_id!r}; skipping",
                    )
                    continue
                text: str = "\n\n".join(
                    c.get("text", "") for c in chunks if c.get("text")
                ).strip()
                if not text:
                    logger.log(
                        "NLP",
                        f"{doc_id}: empty context {u_ctx_id!r}; skipping",
                    )
                    continue
                chunk_id: int = chunks[0].get("chunk_id", -1)
                u_logic_chunk_id: str = chunks[0].get("u_logic_chunk_id", "")
                tokens: int = ctx_entry.get("tokens", 0)
                futures.append((
                    u_ctx_id,
                    u_logic_chunk_id,
                    chunk_id,
                    tokens,
                    pool.submit(
                        extractor.extract,
                        text,
                        doc_id,
                        chunk_id,
                        u_ctx_id,
                    ),
                ))
            for u_ctx_id, u_logic_chunk_id, chunk_id, tokens, fut in futures:
                result: dict = fut.result()
                artifacts.append({
                    "u_ctx_id": u_ctx_id,
                    "u_logic_chunk_id": u_logic_chunk_id,
                    "chunk_id": chunk_id,
                    "tokens": tokens,
                    "extractions": result["extractions"],
                    "chunk_signals": result["chunk_signals"],
                    "errors": result["errors"],
                })

        _write_json({"doc_id": doc_id, "artifacts": artifacts}, out_path)
        logger.log(
            "CHUNK",
            f"{doc_id}: extracted artifacts from {len(artifacts)} contexts -> {out_path}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract pavement-engineering artifacts from mode-3 logical chunks."
    )
    parser.add_argument("--cfg", type=str, default="./extract_artifacts.toml",
                        help="Path to TOML config (default: ./extract_artifacts.toml). "
                             "Also accepts cfg/nemo.toml.")
    parser.add_argument("--input_dir", type=str,
                        help="Override the directory containing *-logic-ctx.json")
    parser.add_argument("--overwrite", action="store_true",
                        help="Force regeneration of cached -logic-artifacts.json")
    parser.add_argument("--host", type=str, default=None,
                        help="Ollama host (Ollama models). Overrides config / $OLLAMA_HOST. "
                             "Cluster slurm pattern: --host localhost --port $PORT")
    parser.add_argument("--port", type=int, default=None, help="Ollama port (Ollama models).")
    args = parser.parse_args()

    cli_ollama_host = None
    if args.host or args.port:
        cli_ollama_host = f"http://{args.host or 'localhost'}:{args.port or 11434}"

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

    main(cfg, overwrite=args.overwrite, ollama_host=cli_ollama_host)
