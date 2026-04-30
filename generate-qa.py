"""
Generate Q&A + citations from random_logical chunker output.

Inputs (per doc, side-by-side in `chunk_dir`):
  - <doc_id>-logic-ctx.json       (contexts; each has chunk text(s))
  - <doc_id>-logic-artifacts.json (per-context extractions + chunk_signals)

Pipeline (mirrors examples/qa-generation/generate-data-async2.py):
  Phase 1 — for each (context, artifact), generate 3-5 questions.
  Phase 2 — for each question, extract a verbatim citation from the context.

Self-contained: no imports from `examples/`. Supports gpt-* (OpenAI structured
outputs), gemini-* (Google structured outputs), and Ollama (any other model).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import tomllib
import traceback
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import loguru
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

from ollama import Client as OllamaClient

# Gemini is loaded lazily — only required if a gemini-* model is selected.
# The example uses the legacy `google.generativeai` SDK; we keep the same path.
try:
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted
    _GEMINI_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore[assignment]

    class ResourceExhausted(Exception):  # type: ignore[no-redef]
        """Stub raised in place of google.api_core.exceptions.ResourceExhausted."""
    _GEMINI_AVAILABLE = False


load_dotenv()

logger = loguru.logger
# Loguru sinks are configured inside main() so the console level can follow
# the --log-level flag while the file sink stays at DEBUG.


# ---------------------------------------------------------------------------
# Pydantic schemas (embedded from examples/response.py)
# ---------------------------------------------------------------------------

class QuestionDifficulty(str, Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate"


class QuestionType(str, Enum):
    FACTUAL = "factual"
    CONCEPTUAL = "conceptual"
    APPLICATION = "application"
    ANALYSIS = "analysis"


class GeneratedQuestion(BaseModel):
    difficulty: QuestionDifficulty
    question: str
    question_type: QuestionType
    answer: str


class GeneratedQuestionsResponse(BaseModel):
    questions: list[GeneratedQuestion]


class CitationResponse(BaseModel):
    citation: str
    first_sentence: str
    last_sentence: str


QUESTIONS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "difficulty": {"type": "string", "enum": ["basic", "intermediate"]},
                    "question": {"type": "string"},
                    "question_type": {
                        "type": "string",
                        "enum": ["factual", "conceptual", "application", "analysis"],
                    },
                    "answer": {"type": "string"},
                },
                "required": ["difficulty", "question", "question_type", "answer"],
            },
        }
    },
    "required": ["questions"],
}

CITATION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "citation": {"type": "string"},
        "first_sentence": {"type": "string"},
        "last_sentence": {"type": "string"},
    },
    "required": ["citation", "first_sentence", "last_sentence"],
}


# ---------------------------------------------------------------------------
# Embedded utils (from examples/utils.py)
# ---------------------------------------------------------------------------

def read_configuration(path: str) -> dict[str, Any]:
    logger.info(f"Reading configuration {path}")
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_to_json(data: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_prompt_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def replace_symbols(text: str, symbols: list[dict]) -> str:
    if not text or not symbols:
        return text
    out = text
    for cfg in symbols:
        values = cfg.get("values", []) or []
        replace_with = cfg.get("replace_with", []) or []
        for i, val in enumerate(values):
            if val and i < len(replace_with):
                out = out.replace(val, replace_with[i])
    return out


# ---------------------------------------------------------------------------
# Model routing (gpt-* / gemini-* / ollama)
# ---------------------------------------------------------------------------

def is_gpt(model: str) -> bool:
    return model.startswith("gpt")


def is_gemini(model: str) -> bool:
    return model.startswith("gemini")


def is_ollama_model(model: str) -> bool:
    return not (is_gpt(model) or is_gemini(model))


def _silent_task_exception_handler(loop, context: dict) -> None:
    """Asyncio exception handler that drops noisy task-shutdown messages.

    On a fatal-error SystemExit, concurrent Tasks finish with their own
    exceptions that are never awaited (the as_completed consumer bailed out
    on the first one). Asyncio's default handler then prints full chained
    tracebacks via Task.__del__. We've already logged the human-readable
    fatal-error reason via _is_fatal_error → SystemExit → main()'s
    final-banner handler; the tracebacks add no signal, so we drop them.
    """
    msg = context.get("message", "") or ""
    if msg.startswith("Task exception was never retrieved"):
        return
    if "Task was destroyed but it is pending" in msg:
        return
    loop.default_exception_handler(context)


def _cancel_all_pending_tasks(loop) -> None:
    """Cancel any tasks still pending on the loop and let cancellation settle.

    Called from main()'s finally blocks before loop.close() so a fatal-error
    SystemExit doesn't leave thousands of pending tasks emitting
    'Task was destroyed but it is pending!' warnings on shutdown. Also drains
    exceptions from already-done tasks so concurrent failures don't surface as
    'Task exception was never retrieved' tracebacks.
    """
    all_t = list(asyncio.all_tasks(loop))
    pending = [t for t in all_t if not t.done()]
    if pending:
        for t in pending:
            t.cancel()
        try:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except BaseException:
            pass
    for t in all_t:
        if t.done() and not t.cancelled():
            try:
                t.exception()  # consume to silence "never retrieved" warnings
            except (asyncio.CancelledError, asyncio.InvalidStateError):
                pass


def _is_fatal_error(exc: Exception, model: str) -> bool:
    """Permanent provider errors that no retry will fix.

    Authentication, authorization, malformed request, missing model.
    Quota (ResourceExhausted) is already handled in its own except branch.
    """
    if is_gemini(model) and _GEMINI_AVAILABLE:
        from google.api_core import exceptions as gax
        if isinstance(exc, (
            gax.InvalidArgument,     # 400 — bad key, bad schema, bad payload
            gax.Unauthenticated,     # 401
            gax.PermissionDenied,    # 403
            gax.NotFound,            # 404 — wrong model name / endpoint
            gax.FailedPrecondition,  # config-level rejections
        )):
            return True
    if is_gpt(model):
        try:
            from openai import (
                AuthenticationError,
                PermissionDeniedError,
                NotFoundError,
                BadRequestError,
            )
        except ImportError:
            pass
        else:
            if isinstance(exc, (
                AuthenticationError,
                PermissionDeniedError,
                NotFoundError,
                BadRequestError,
            )):
                return True
    if is_ollama_model(model):
        msg = str(exc).lower()
        if any(s in msg for s in (
            "connection refused",
            "model not found",
            "no such model",
        )):
            return True
    return False


def initialize_client(model: str, host: str = "localhost", port: int = 11434):
    if is_gpt(model):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        client = OpenAI(api_key=api_key, timeout=300)
        logger.info(f"Initialized OpenAI client for {model}")
        return client
    if is_gemini(model):
        if not _GEMINI_AVAILABLE:
            raise RuntimeError(
                "Gemini support requires `google-generativeai`. "
                "Install with `pip install google-generativeai`."
            )
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY_V13")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")
        genai.configure(api_key=api_key)
        client = genai.GenerativeModel(model)
        logger.info(f"Initialized Gemini client for {model}")
        return client
    client = OllamaClient(host=f"http://{host}:{port}")
    logger.info(f"Initialized Ollama client at http://{host}:{port} for {model}")
    return client


# ---------------------------------------------------------------------------
# Inference wrappers (gpt-* / gemini-* / ollama)
# ---------------------------------------------------------------------------

def query_openai_structured(client, prompt: str, model: str, json_schema, temperature: float = 0.0):
    response = client.responses.parse(
        model=model,
        input=[{"role": "user", "content": prompt}],
        text_format=json_schema,
        temperature=temperature,
    )
    return response


def query_gemini_structured(client, prompt: str, model: str, response_schema: dict, temperature: float = 0.0):
    generation_config = {
        "temperature": temperature,
        "response_mime_type": "application/json",
        "response_schema": response_schema,
    }
    contents = [{"role": "user", "parts": [prompt]}]
    return client.generate_content(
        contents,
        generation_config=generation_config,
        request_options={"timeout": 300},
    )


def query_ollama_structured(client, prompt: str, model: str, format_schema: dict, temperature: float = 0.0):
    return client.chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        options={"temperature": temperature},
        format=format_schema,
        think=False,
    )


# ---------------------------------------------------------------------------
# Doc ingestion
# ---------------------------------------------------------------------------

CTX_SUFFIX = "-logic-ctx.json"
ART_SUFFIX = "-logic-artifacts.json"


@dataclass
class QATask:
    """A single (context, artifact) unit for the QA generation phase."""
    doc_id: str
    u_ctx_id: str
    u_logic_chunk_id: str
    source_u_chunk_ids: list[str]
    context_text: str
    doc_info: str
    artifact_category: str
    artifact_id: str
    u_artifact_id: str
    artifact: dict   # raw artifact dict for output

    @property
    def task_key(self) -> str:
        return f"{self.u_ctx_id}::{self.artifact_id}"


def iter_doc_pairs(chunk_dir: str) -> list[tuple[str, str, str]]:
    base = Path(chunk_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"chunk_dir does not exist: {chunk_dir}")

    pairs: list[tuple[str, str, str]] = []
    for ctx_path in sorted(base.glob(f"*{CTX_SUFFIX}")):
        # The ctx file's top-level "doc_id" field is the authoritative identifier
        # written by `_nemo.py::_build_logical_contexts`. Trust it; fall back to
        # filename derivation only if missing/empty. Soft-warn on mismatch — usually
        # signals a rename or relocation drift.
        ctx_data = load_json(str(ctx_path))
        filename_id = ctx_path.name[: -len(CTX_SUFFIX)]
        json_id = (ctx_data.get("doc_id") or "").strip() if isinstance(ctx_data, dict) else ""
        doc_id = json_id or filename_id
        if json_id and json_id != filename_id:
            logger.warning(
                f"{ctx_path}: doc_id mismatch — JSON says {json_id!r}, "
                f"filename derives {filename_id!r}; using JSON field"
            )
        art_path = base / f"{doc_id}{ART_SUFFIX}"
        if not art_path.exists():
            logger.warning(f"Missing artifacts file for {doc_id}: {art_path}")
            continue
        pairs.append((doc_id, str(ctx_path), str(art_path)))
    return pairs


def build_context_text(ctx_entry: dict) -> str:
    """Concatenate chunk texts for a single context entry."""
    parts = [c.get("text", "") for c in ctx_entry.get("chunks", []) if c.get("text")]
    return "\n\n".join(parts).strip()


def build_doc_info(doc_id: str, artifact_entry: dict) -> str:
    """Compose document/context info from chunk_signals (summary, scope, topics)."""
    sig = artifact_entry.get("chunk_signals", {}) or {}
    summary_obj = sig.get("summary", {}) or {}
    summary = summary_obj.get("summary", "")
    scope = summary_obj.get("scope", "")
    doc_funcs = summary_obj.get("document_functions", []) or []
    topics = sig.get("topics", []) or []

    lines = [f"Document ID: {doc_id}"]
    if summary:
        lines.append(f"Summary: {summary}")
    if scope:
        lines.append(f"Scope: {scope}")
    if doc_funcs:
        lines.append(f"Document Functions: {', '.join(doc_funcs)}")
    if topics:
        topic_strs = [
            f"{t.get('topic', '')} ({t.get('role', '')})".strip()
            for t in topics if t.get("topic")
        ]
        if topic_strs:
            lines.append(f"Topics: {'; '.join(topic_strs)}")
    return "\n".join(lines)


def format_artifact(category: str, artifact: dict) -> str:
    """Pretty-print an artifact for {ARTIFACT} substitution."""
    parts = [f"Category: {category}"]
    text = artifact.get("text", "")
    if text:
        parts.append(f"Text: {text}")
    desc = artifact.get("description", "")
    if desc:
        parts.append(f"Description: {desc}")
    sig = artifact.get("significance")
    if sig:
        parts.append(f"Significance: {sig}")
    attrs = artifact.get("attributes", {}) or {}
    if attrs:
        parts.append("Attributes:")
        for k, v in attrs.items():
            parts.append(f"  - {k}: {v}")
    return "\n".join(parts)


def build_summary_artifact(artifact_entry: dict) -> Optional[dict]:
    """Construct a synthetic 'summary' artifact from chunk_signals.summary."""
    sig = artifact_entry.get("chunk_signals", {}) or {}
    summary_obj = sig.get("summary", {}) or {}
    summary_text = summary_obj.get("summary", "")
    if not summary_text:
        return None
    u_ctx_id = artifact_entry.get("u_ctx_id", "")
    return {
        "artifact_id": f"{u_ctx_id}_summary",
        "u_artifact_id": f"{u_ctx_id}-summary",
        "text": summary_text,
        "description": "Context-level summary",
        "significance": None,
        "attributes": {
            "scope": summary_obj.get("scope", ""),
            "document_functions": summary_obj.get("document_functions", []) or [],
        },
    }


def build_tasks(
    chunk_dir: str,
    artifact_categories: list[str],
    include_summary: bool,
    max_artifacts_per_ctx: int,
) -> list[QATask]:
    tasks: list[QATask] = []
    for doc_id, ctx_path, art_path in iter_doc_pairs(chunk_dir):
        ctx_data = load_json(ctx_path)
        art_data = load_json(art_path)

        # Cross-check: artifacts file's top-level doc_id (written by
        # extract_artifacts.py v6+) should match the resolved doc_id. Soft-warn
        # on mismatch — usually signals a rename or stale cache.
        art_doc_id = (art_data.get("doc_id") or "").strip() if isinstance(art_data, dict) else ""
        if art_doc_id and art_doc_id != doc_id:
            logger.warning(
                f"{art_path}: doc_id mismatch — artifacts file says {art_doc_id!r}, "
                f"expected {doc_id!r} (from ctx file); proceeding with ctx doc_id"
            )

        # Index artifact entries by u_ctx_id
        art_by_ctx: dict[str, dict] = {}
        for entry in art_data.get("artifacts", []) or []:
            uid = entry.get("u_ctx_id")
            if uid:
                art_by_ctx[uid] = entry

        for ctx_entry in ctx_data.get("contexts", []) or []:
            u_ctx_id = ctx_entry.get("u_ctx_id", "")
            if not u_ctx_id:
                continue

            context_text = build_context_text(ctx_entry)
            if not context_text:
                logger.warning(f"Empty context for {u_ctx_id}; skipping")
                continue

            chunks = ctx_entry.get("chunks", []) or []
            u_logic_chunk_id = chunks[0].get("u_logic_chunk_id", "") if chunks else ""
            source_u_chunk_ids: list[str] = []
            for c in chunks:
                source_u_chunk_ids.extend(c.get("source_u_chunk_ids", []) or [])

            art_entry = art_by_ctx.get(u_ctx_id, {})
            doc_info = build_doc_info(doc_id, art_entry)

            # Flatten extractions across configured categories
            extractions = art_entry.get("extractions", {}) or {}
            ctx_artifacts: list[tuple[str, dict]] = []
            for category in artifact_categories:
                for art in extractions.get(category, []) or []:
                    ctx_artifacts.append((category, art))

            # Optional cap
            if max_artifacts_per_ctx and len(ctx_artifacts) > max_artifacts_per_ctx:
                ctx_artifacts = ctx_artifacts[:max_artifacts_per_ctx]

            # Optional synthetic summary element
            if include_summary:
                summary_art = build_summary_artifact(art_entry)
                if summary_art is not None:
                    ctx_artifacts.append(("summary", summary_art))

            for category, art in ctx_artifacts:
                tasks.append(QATask(
                    doc_id=doc_id,
                    u_ctx_id=u_ctx_id,
                    u_logic_chunk_id=u_logic_chunk_id,
                    source_u_chunk_ids=source_u_chunk_ids,
                    context_text=context_text,
                    doc_info=doc_info,
                    artifact_category=category,
                    artifact_id=art.get("artifact_id", ""),
                    u_artifact_id=art.get("u_artifact_id", ""),
                    artifact=art,
                ))

    return tasks


# ---------------------------------------------------------------------------
# Phase 1 — QA generation
# ---------------------------------------------------------------------------

async def generate_qa_for_task_async(
    task: QATask,
    template: str,
    client,
    model: str,
    symbol_cfg: Optional[dict],
) -> tuple[QATask, list[dict], bool, Optional[str]]:
    ctx_text = task.context_text
    if symbol_cfg and symbol_cfg.get("enabled"):
        ctx_text = replace_symbols(ctx_text, symbol_cfg.get("symbols", []))

    artifact_block = format_artifact(task.artifact_category, task.artifact)

    prompt = (
        template
        .replace("{CONTEXT}", ctx_text)
        .replace("{DOCUMENT_INFO}", task.doc_info)
        .replace("{ARTIFACT_CATEGORY}", task.artifact_category)
        .replace("{ARTIFACT}", artifact_block)
    )

    max_tries = 5
    for retry in range(max_tries):
        try:
            loop = asyncio.get_event_loop()
            if is_gpt(model):
                response = await loop.run_in_executor(
                    None,
                    lambda: query_openai_structured(client, prompt, model, GeneratedQuestionsResponse),
                )
                gen = response.output_parsed
            elif is_gemini(model):
                response = await loop.run_in_executor(
                    None,
                    lambda: query_gemini_structured(client, prompt, model, QUESTIONS_SCHEMA),
                )
                gen = GeneratedQuestionsResponse.model_validate_json(response.text)
            else:
                response = await loop.run_in_executor(
                    None,
                    lambda: query_ollama_structured(
                        client, prompt, model, GeneratedQuestionsResponse.model_json_schema()
                    ),
                )
                gen = GeneratedQuestionsResponse.model_validate_json(
                    response.get("message", {}).get("content", "")
                )

            questions: list[dict] = []
            for q_idx, q in enumerate(gen.questions):
                logger.debug(f"Q[{task.task_key}#{q_idx}]: {q.question}")
                questions.append({
                    "question_id": f"TEMP-{task.u_ctx_id}-{task.artifact_id}-{q_idx}",
                    "question": q.question,
                    "answer": q.answer,
                    "full_citation": None,
                    "question_type": q.question_type.value if hasattr(q.question_type, "value") else str(q.question_type),
                    "question_difficulty": q.difficulty.value if hasattr(q.difficulty, "value") else str(q.difficulty),
                    "question_element_type": task.artifact_category,
                    "doc_id": task.doc_id,
                    "u_ctx_id": task.u_ctx_id,
                    "u_logic_chunk_id": task.u_logic_chunk_id,
                    "source_u_chunk_ids": task.source_u_chunk_ids,
                    "artifact_id": task.artifact_id,
                    "u_artifact_id": task.u_artifact_id,
                    "artifact": task.artifact,
                    "context_text": task.context_text,
                    "model_qa": model,
                    "model_citation": None,
                    "citation_extracted": False,
                })
            return task, questions, True, None

        except ResourceExhausted as e:
            logger.error(f"❌ QUOTA EXCEEDED: {e}")
            raise SystemExit(f"API quota exceeded: {e}")
        except Exception as e:
            if _is_fatal_error(e, model):
                logger.error(
                    f"❌ FATAL provider error on task {task.task_key} "
                    f"(no retry — verify API key / model / config): {e}"
                )
                raise SystemExit(f"Fatal provider error: {e}")
            logger.error(f"Error on task {task.task_key} (retry {retry + 1}/{max_tries}): {e}\n{traceback.format_exc()}")
            if retry == max_tries - 1:
                return task, [], False, str(e)

    return task, [], False, "Unknown error"


async def run_phase1(
    tasks: list[QATask],
    template: str,
    client,
    model: str,
    max_concurrent: int,
    symbol_cfg: Optional[dict],
    qa_intermediate_path: str,
    save_every: int,
    existing_questions: list[dict],
) -> list[dict]:
    done_keys: set[str] = set()
    for q in existing_questions:
        key = f"{q.get('u_ctx_id', '')}::{q.get('artifact_id', '')}"
        done_keys.add(key)

    pending = [t for t in tasks if t.task_key not in done_keys]
    logger.info(f"Phase 1: {len(pending)} pending / {len(tasks)} total tasks "
                f"({len(done_keys)} (ctx, artifact) pairs already done)")

    if not pending:
        return existing_questions

    semaphore = asyncio.Semaphore(max_concurrent)
    all_questions = list(existing_questions)

    async def bounded(task: QATask):
        async with semaphore:
            return await generate_qa_for_task_async(task, template, client, model, symbol_cfg)

    coros = [bounded(t) for t in pending]
    completed = 0
    with tqdm(total=len(coros), desc="Phase 1: QA", unit="task") as pbar:
        for fut in asyncio.as_completed(coros):
            task, questions, success, err = await fut
            if success:
                all_questions.extend(questions)
                logger.debug(f"✓ {task.task_key}: {len(questions)} questions")
            else:
                logger.error(f"✗ {task.task_key}: {err}")
            completed += 1
            pbar.update(1)
            if save_every and completed % save_every == 0:
                save_to_json(all_questions, qa_intermediate_path)
                logger.info(f"Phase 1 periodic save: {completed}/{len(coros)} -> {qa_intermediate_path}")

    save_to_json(all_questions, qa_intermediate_path)
    return all_questions


# ---------------------------------------------------------------------------
# Phase 2 — citation extraction
# ---------------------------------------------------------------------------

async def extract_citation_for_question_async(
    question_text: str,
    context_text: str,
    template: str,
    client,
    model: str,
    symbol_cfg: Optional[dict],
) -> CitationResponse:
    ctx = context_text
    if symbol_cfg and symbol_cfg.get("enabled"):
        ctx = replace_symbols(ctx, symbol_cfg.get("symbols", []))

    prompt = template.replace("{QUESTION}", question_text).replace("{CONTEXT}", ctx)

    max_tries = 5
    for retry in range(max_tries):
        try:
            loop = asyncio.get_event_loop()
            if is_gpt(model):
                response = await loop.run_in_executor(
                    None,
                    lambda: query_openai_structured(client, prompt, model, CitationResponse),
                )
                citation = response.output_parsed
            elif is_gemini(model):
                response = await loop.run_in_executor(
                    None,
                    lambda: query_gemini_structured(client, prompt, model, CITATION_SCHEMA),
                )
                citation = CitationResponse.model_validate_json(response.text)
            else:
                response = await loop.run_in_executor(
                    None,
                    lambda: query_ollama_structured(
                        client, prompt, model, CitationResponse.model_json_schema()
                    ),
                )
                citation = CitationResponse.model_validate_json(
                    response.get("message", {}).get("content", "")
                )

            logger.debug(f"Citation extracted: {citation.citation[:80]}...")
            return citation

        except ResourceExhausted as e:
            logger.error(f"❌ QUOTA EXCEEDED: {e}")
            raise SystemExit(f"API quota exceeded: {e}")
        except Exception as e:
            if _is_fatal_error(e, model):
                logger.error(
                    f"❌ FATAL provider error during citation extraction "
                    f"(no retry — verify API key / model / config): {e}"
                )
                raise SystemExit(f"Fatal provider error: {e}")
            logger.error(f"Citation error (retry {retry + 1}/{max_tries}): {e}\n{traceback.format_exc()}")

    return CitationResponse(
        citation="Max retries reached. No citation available.",
        first_sentence="Max retries reached. No citation available.",
        last_sentence="Max retries reached. No citation available.",
    )


async def run_phase2(
    questions: list[dict],
    template: str,
    client,
    model: str,
    max_concurrent: int,
    symbol_cfg: Optional[dict],
    citations_intermediate_path: str,
    save_every: int,
) -> list[dict]:
    # Resume: merge any prior citations from intermediate file
    if os.path.exists(citations_intermediate_path):
        try:
            prior = load_json(citations_intermediate_path)
            citations_map = {q["question_id"]: q for q in prior if q.get("citation_extracted")}
            for q in questions:
                qid = q.get("question_id")
                if qid in citations_map:
                    q["full_citation"] = citations_map[qid]["full_citation"]
                    q["citation_extracted"] = True
                    q["model_citation"] = citations_map[qid].get("model_citation", model)
            logger.info(f"Phase 2 resume: {len(citations_map)} questions already have citations")
        except Exception as e:
            logger.warning(f"Could not resume from {citations_intermediate_path}: {e}")

    pending = [q for q in questions if not q.get("citation_extracted")]
    logger.info(f"Phase 2: {len(pending)} pending / {len(questions)} total questions")

    if not pending:
        return questions

    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded(q: dict):
        async with semaphore:
            citation = await extract_citation_for_question_async(
                q["question"], q.get("context_text", ""), template, client, model, symbol_cfg
            )
            q["full_citation"] = citation.model_dump()
            q["citation_extracted"] = True
            q["model_citation"] = model
            return q

    coros = [bounded(q) for q in pending]
    completed = 0
    with tqdm(total=len(coros), desc="Phase 2: Citations", unit="q") as pbar:
        for fut in asyncio.as_completed(coros):
            await fut
            completed += 1
            pbar.update(1)
            if save_every and completed % save_every == 0:
                save_to_json(questions, citations_intermediate_path)
                logger.info(f"Phase 2 periodic save: {completed}/{len(coros)} -> {citations_intermediate_path}")

    save_to_json(questions, citations_intermediate_path)
    return questions


# ---------------------------------------------------------------------------
# Output: CSV
# ---------------------------------------------------------------------------

def save_questions_to_csv(questions: list[dict], path: str) -> None:
    rows = []
    for idx, q in enumerate(questions):
        rows.append({
            "pandas_index": idx,
            "question_number": idx,
            "question_id": q.get("question_id", ""),
            "question": q.get("question", ""),
            "answer": q.get("answer", ""),
            "citation": (q.get("full_citation") or {}).get("citation", ""),
            "question_type": q.get("question_type", ""),
            "question_element_type": q.get("question_element_type", ""),
            "question_difficulty": q.get("question_difficulty", ""),
            "doc_id": q.get("doc_id", ""),
            "u_ctx_id": q.get("u_ctx_id", ""),
            "u_logic_chunk_id": q.get("u_logic_chunk_id", ""),
            "source_u_chunk_ids": str(q.get("source_u_chunk_ids", [])),
            "artifact_id": q.get("artifact_id", ""),
            "u_artifact_id": q.get("u_artifact_id", ""),
            "artifact": str(q.get("artifact", {})),
            "full_citation": q.get("full_citation", ""),
            "model_qa": q.get("model_qa", ""),
            "model_citation": q.get("model_citation", ""),
        })
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    logger.info(f"CSV saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate Q&A + citations from logical-chunk artifacts")
    parser.add_argument("--config", default="generate-qa.toml", help="Path to TOML config")
    parser.add_argument("--host", default="localhost", help="Ollama host")
    parser.add_argument("--port", type=int, default=11434, help="Ollama port")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console (stderr) log level. File log stays at DEBUG. "
             "Use WARNING for clean progress-bar output.",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level)
    os.makedirs("logs", exist_ok=True)
    logger.add("logs/generate-qa.log", rotation="5 MB", compression="zip", level="DEBUG")

    cfg_root = read_configuration(args.config)
    cfg = cfg_root["generate-qa"]

    chunk_dir: str = cfg["chunk_dir"]
    output_dir: str = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    output_file = cfg["output_file"]
    output_qa_file = cfg.get("output_qa_file") or f"{Path(output_file).stem}_qa_only{Path(output_file).suffix}"
    output_csv_file = cfg["output_csv_file"]

    qa_template = load_prompt_from_file(cfg["question_generate_prompt"])
    citation_template = load_prompt_from_file(cfg["extract_citation_prompt"])

    model_qa: str = cfg["model_qa"]
    model_citations: str = cfg["model_citations"]

    max_concurrent_qa = int(cfg.get("max_concurrent_qa", 5))
    max_concurrent_citations = int(cfg.get("max_concurrent_citations", 5))
    save_every_qa = int(cfg.get("periodic_save_interval_qa", 10))
    save_every_cit = int(cfg.get("periodic_save_interval_citations", 50))

    artifact_categories = list(cfg.get("artifact_categories", []) or [])
    include_summary = bool(cfg.get("include_summary_element", True))
    max_artifacts_per_ctx = int(cfg.get("max_artifacts_per_ctx", 0) or 0)

    symbol_cfg = {
        "enabled": bool(cfg.get("replace_symbols", False)),
        "symbols": cfg.get("symbols", []) or [],
    }
    if symbol_cfg["enabled"]:
        logger.info(f"Symbol replacement enabled: {len(symbol_cfg['symbols'])} group(s)")

    do_qa = bool(cfg.get("generate_qa", True))
    do_cit = bool(cfg.get("extract_citations", True))
    logger.info(f"Phase control: generate_qa={do_qa}, extract_citations={do_cit}")

    # Build tasks from chunk_dir
    tasks = build_tasks(chunk_dir, artifact_categories, include_summary, max_artifacts_per_ctx)
    logger.info(f"Built {len(tasks)} (context, artifact) tasks from {chunk_dir}")
    if not tasks:
        logger.warning("No tasks to process. Exiting.")
        return

    # Initialize clients (separate for qa / citations; same client when models match)
    client_qa = initialize_client(model_qa, host=args.host, port=args.port)
    if model_citations == model_qa:
        client_cit = client_qa
    else:
        client_cit = initialize_client(model_citations, host=args.host, port=args.port)

    qa_intermediate_path = os.path.join(output_dir, output_qa_file)
    base_name, ext = os.path.splitext(output_file)
    citations_intermediate_path = os.path.join(output_dir, f"{base_name}_with_citations{ext}")

    # ---- PHASE 1 ----
    all_questions: list[dict] = []
    if do_qa:
        logger.info("=" * 60)
        logger.info("PHASE 1: Q&A Generation")
        logger.info("=" * 60)
        existing: list[dict] = []
        if os.path.exists(qa_intermediate_path):
            try:
                existing = load_json(qa_intermediate_path)
                logger.info(f"Found existing Q&A intermediate: {len(existing)} questions")
            except Exception as e:
                logger.warning(f"Could not load existing intermediate: {e}")

        t0 = time.time()
        loop = asyncio.new_event_loop()
        loop.set_exception_handler(_silent_task_exception_handler)
        try:
            asyncio.set_event_loop(loop)
            all_questions = loop.run_until_complete(run_phase1(
                tasks, qa_template, client_qa, model_qa,
                max_concurrent_qa, symbol_cfg,
                qa_intermediate_path, save_every_qa, existing,
            ))
        finally:
            _cancel_all_pending_tasks(loop)
            loop.close()
        logger.info(f"Phase 1 complete: {len(all_questions)} questions in {time.time() - t0:.1f}s")
    else:
        logger.info("PHASE 1 skipped (generate_qa=false)")
        if os.path.exists(qa_intermediate_path):
            all_questions = load_json(qa_intermediate_path)
            logger.info(f"Loaded {len(all_questions)} questions from {qa_intermediate_path}")
        else:
            logger.warning(f"No intermediate Q&A file at {qa_intermediate_path}; nothing to do")
            return

    # ---- PHASE 2 ----
    if do_cit:
        logger.info("=" * 60)
        logger.info("PHASE 2: Citation Extraction")
        logger.info("=" * 60)
        t0 = time.time()
        loop = asyncio.new_event_loop()
        loop.set_exception_handler(_silent_task_exception_handler)
        try:
            asyncio.set_event_loop(loop)
            all_questions = loop.run_until_complete(run_phase2(
                all_questions, citation_template, client_cit, model_citations,
                max_concurrent_citations, symbol_cfg,
                citations_intermediate_path, save_every_cit,
            ))
        finally:
            _cancel_all_pending_tasks(loop)
            loop.close()
        logger.info(f"Phase 2 complete in {time.time() - t0:.1f}s")
    else:
        logger.info("PHASE 2 skipped (extract_citations=false)")

    # ---- FINAL: sort to source order, renumber, save ----
    def _natural_key(s: str) -> tuple:
        # split on digit runs so "ctx-2" sorts before "ctx-10"
        import re
        return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", s or ""))

    all_questions.sort(key=lambda q: (
        _natural_key(q.get("doc_id", "")),
        _natural_key(q.get("u_ctx_id", "")),
        _natural_key(q.get("artifact_id", "")),
    ))
    counters: dict[str, int] = {}
    for q in all_questions:
        ctx_id = q.get("u_ctx_id", "unknown")
        counters[ctx_id] = counters.get(ctx_id, 0)
        q["question_id"] = f"{ctx_id}-q-{counters[ctx_id]}"
        counters[ctx_id] += 1
        q.pop("citation_extracted", None)
        q.pop("citation", None)  # strip legacy top-level field from old intermediates

    out_path = os.path.join(output_dir, output_file)
    save_to_json(all_questions, out_path)
    logger.info(f"Saved {len(all_questions)} questions to {out_path}")

    base_name, ext = os.path.splitext(output_file)
    wo_context_path = os.path.join(output_dir, f"{base_name}_wo_context{ext}")
    wo_context = [{k: v for k, v in q.items() if k != "context_text"} for q in all_questions]
    save_to_json(wo_context, wo_context_path)
    logger.info(f"Saved {len(wo_context)} questions (without context_text) to {wo_context_path}")

    save_questions_to_csv(all_questions, os.path.join(output_dir, output_csv_file))


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        # Convert our fatal-error SystemExit (raised with a message string from
        # _is_fatal_error / quota paths) into a clean final banner so the
        # reason is visible at the bottom of the output without scrolling up.
        # Numeric exit codes (e.g. argparse --help exits 0) pass through unchanged.
        if isinstance(e.code, str):
            logger.error("=" * 60)
            logger.error("❌ Execution stopped")
            logger.error(f"Reason: {e.code}")
            logger.error("=" * 60)
            sys.exit(1)
        raise
