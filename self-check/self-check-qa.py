"""Self-check QA evaluation for generate-qa.py output.

LLM-as-judge: for each (question, answer, context_text) record produced by
generate-qa.py, ask an Ollama-served model whether the answer is supported
by the context with respect to the question. Output a 3-valued score
(0 / 0.5 / 1) per record, with -1 reserved for records that have no
context_text and are therefore skipped without an LLM call.

Self-contained: does not depend on the sibling `utils`/`inference`/`response`
modules that the `self-check/example/` script imports.
"""

import argparse
import asyncio
import copy
import json
import os
import sys
from typing import Any, Dict, List, Optional

import loguru
from dotenv import load_dotenv
from ollama import AsyncClient
from pydantic import BaseModel, ConfigDict, Field, field_validator
from tqdm.asyncio import tqdm as atqdm

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

load_dotenv()

logger = loguru.logger
os.makedirs("logs", exist_ok=True)
logger.add(
    "logs/self-check-qa.log",
    rotation="5 MB",
    compression="zip",
    level="DEBUG",
)


# ============================================================
# Inline helpers (the example pulls these from a sibling project)
# ============================================================


def read_configuration(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_to_json(obj: Any, path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_prompt_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def send_pushover_message(
    title: str,
    message: str,
    priority: int = 0,
    sound: str = "magic",
    url: Optional[str] = None,
    url_title: Optional[str] = None,
) -> None:
    """Best-effort Pushover notification. No-op when credentials are absent."""
    user = os.getenv("PUSHOVER_USER")
    token = os.getenv("PUSHOVER_TOKEN")
    if not user or not token:
        logger.warning(
            "PUSHOVER_USER / PUSHOVER_TOKEN not set — skipping Pushover notification."
        )
        return
    try:
        import requests
    except ImportError:
        logger.warning("`requests` not installed — skipping Pushover notification.")
        return

    payload = {
        "token": token,
        "user": user,
        "title": title,
        "message": message,
        "priority": priority,
        "sound": sound,
    }
    if url:
        payload["url"] = url
    if url_title:
        payload["url_title"] = url_title
    try:
        resp = requests.post(
            "https://api.pushover.net/1/messages.json", data=payload, timeout=10
        )
        if resp.status_code != 200:
            logger.warning(
                f"Pushover returned status {resp.status_code}: {resp.text}"
            )
    except Exception as e:
        logger.warning(f"Pushover request failed: {e}")


# ============================================================
# Response schema — strict 0 / 0.5 / 1 evaluation
# ============================================================


class QuestionAnswerEvaluationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation: float = Field(description="0 if No, 0.5 if N/A, 1 if Yes")

    @field_validator("evaluation")
    @classmethod
    def validate_evaluation(cls, value: float) -> float:
        if value not in (0, 0.5, 1):
            raise ValueError("evaluation must be one of: 0, 0.5, 1")
        return value


# ============================================================
# Async Ollama wrapper
# ============================================================


async def query_ollama_chat_async(
    client: AsyncClient,
    prompt: str,
    model: str,
    *,
    temperature: float = 0.0,
    system: Optional[str] = None,
    format: Optional[Dict] = None,
    think: bool = False,
) -> Dict[str, Any]:
    """Mirrors the example's query_ollama_chat2_async."""
    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    options: Dict[str, Any] = {"temperature": temperature}
    response = await client.chat(
        messages=messages,
        model=model,
        options=options,
        format=format,
        think=think,
    )
    return response


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Self-check evaluation of generate-qa.py output via Ollama."
    )
    parser.add_argument(
        "--config",
        dest="config",
        type=str,
        default="./self-check-qa.toml",
        help=(
            "path to the TOML config (default: ./self-check-qa.toml — assumes "
            "you run this script from the self-check/ directory)"
        ),
    )
    parser.add_argument(
        "--host",
        dest="host",
        type=str,
        default="localhost",
        help="Ollama host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        dest="port",
        type=int,
        default=11434,
        help="Ollama port (default: 11434)",
    )
    parser.add_argument(
        "--limit",
        dest="limit",
        type=int,
        default=None,
        help=(
            "Debug: evaluate only the first N records from input_qa_json. "
            "Applied before resume/skip logic, so outputs will contain at "
            "most N entries. Omit for a full run."
        ),
    )
    args = parser.parse_args()

    config = read_configuration(args.config)
    cfg = config["self-check-qa"]

    input_qa_json = cfg["input_qa_json"]
    prompt_eval_path = cfg["prompt_eval"]
    model = cfg["model"]
    output_json_path = cfg["output_json"]
    output_json_wo_context_path = cfg["output_json_wo_context"]
    output_csv_path = cfg["output_csv"]
    intermediate_json_path = cfg["intermediate_json"]
    retries_number = cfg.get("retries_number", 3)
    periodic_save_interval = cfg.get("periodic_save_interval", 200)
    max_concurrent_questions = cfg.get("max_concurrent_questions", 1)
    debug = cfg.get("debug", False)

    logger.info(f"Loading input QA records from {input_qa_json}")
    qa_records: List[Dict[str, Any]] = load_json(input_qa_json)
    if not isinstance(qa_records, list):
        logger.error(
            f"Expected a JSON list in {input_qa_json}, got {type(qa_records).__name__}"
        )
        sys.exit(1)

    logger.info(f"Loaded {len(qa_records)} records.")

    if args.limit is not None:
        if args.limit < 0:
            logger.error(f"--limit must be non-negative, got {args.limit}")
            sys.exit(1)
        original_count = len(qa_records)
        qa_records = qa_records[: args.limit]
        logger.info(
            f"--limit {args.limit} applied: evaluating {len(qa_records)} / "
            f"{original_count} records (debug mode)."
        )

    # Authoritative input order (matches CSV row order at the end).
    input_order_ids = [r["question_id"] for r in qa_records]
    questions_lookup: Dict[str, Dict[str, Any]] = {
        r["question_id"]: r for r in qa_records
    }

    prompt_template = load_prompt_from_file(prompt_eval_path)
    logger.info(f"Loaded prompt template from {prompt_eval_path}")

    async_client = AsyncClient(host=f"http://{args.host}:{args.port}")

    # ---- Resume: load intermediate (if present) else completed output ----
    processed_ids: set = set()
    if os.path.exists(intermediate_json_path):
        try:
            prev = load_json(intermediate_json_path)
            for q in prev:
                qid = q.get("question_id")
                if qid in questions_lookup:
                    ea = q.get("question_evaluation", {}).get("evaluation_answering")
                    if ea is not None:
                        questions_lookup[qid].setdefault("question_evaluation", {})
                        questions_lookup[qid]["question_evaluation"][
                            "evaluation_answering"
                        ] = ea
                        processed_ids.add(qid)
            logger.info(
                f"Resuming from intermediate file: {len(processed_ids)} records "
                f"already processed."
            )
        except Exception as e:
            logger.warning(
                f"Failed to load intermediate {intermediate_json_path}: {e}. "
                "Starting fresh."
            )
            processed_ids = set()
    elif os.path.exists(output_json_path):
        try:
            prev = load_json(output_json_path)
            for q in prev:
                qid = q.get("question_id")
                if qid in questions_lookup:
                    ea = q.get("question_evaluation", {}).get("evaluation_answering")
                    if ea is not None:
                        questions_lookup[qid].setdefault("question_evaluation", {})
                        questions_lookup[qid]["question_evaluation"][
                            "evaluation_answering"
                        ] = ea
                        processed_ids.add(qid)
            logger.info(
                f"No intermediate file; loaded {len(processed_ids)} previously "
                f"completed records from {output_json_path}."
            )
        except Exception as e:
            logger.warning(
                f"Failed to load output {output_json_path}: {e}. Starting fresh."
            )
            processed_ids = set()
    else:
        logger.info("No prior results found — starting fresh.")

    # ---- Build task list of records still to process ----
    pending = [
        qid for qid in input_order_ids if qid not in processed_ids
    ]
    logger.info(
        f"To process: {len(pending)} / {len(input_order_ids)} records "
        f"(skipping {len(processed_ids)} already-processed)."
    )

    semaphore = asyncio.Semaphore(max_concurrent_questions)

    async def process_one(question_id: str) -> Optional[Dict[str, Any]]:
        async with semaphore:
            record = questions_lookup[question_id]
            question_text = record.get("question", "") or ""
            answer_text = record.get("answer", "") or ""
            context_text = record.get("context_text") or ""

            if not str(context_text).strip():
                logger.debug(
                    f"Skipping LLM for question_id={question_id}: context_text is empty."
                )
                return {
                    "question_id": question_id,
                    "model": model,
                    "evaluation": -1,
                }

            prompt = (
                prompt_template
                .replace("{QUESTION}", str(question_text))
                .replace("{ANSWER}", str(answer_text))
                .replace("{PROCESS_CITATION}", str(context_text))
            )

            llm_result: Optional[QuestionAnswerEvaluationResponse] = None
            full_response: Any = None
            for attempt in range(1, retries_number + 1):
                try:
                    response = await query_ollama_chat_async(
                        async_client,
                        prompt,
                        model=model,
                        temperature=0.0,
                        format=QuestionAnswerEvaluationResponse.model_json_schema(),
                        think=False,
                    )
                    response_text = (
                        response.get("message", {}).get("content", "")
                        if isinstance(response, dict)
                        else getattr(response, "message", {}).get("content", "")
                    )
                    llm_result = QuestionAnswerEvaluationResponse.model_validate_json(
                        response_text
                    )
                    full_response = response
                    break
                except Exception as e:
                    logger.warning(
                        f"Retry {attempt}/{retries_number} for "
                        f"question_id={question_id}: {e}"
                    )

            if llm_result is None:
                logger.error(
                    f"Failed to evaluate question_id={question_id} after "
                    f"{retries_number} retries."
                )
                return None

            result: Dict[str, Any] = {
                "question_id": question_id,
                "model": model,
                "evaluation": llm_result.evaluation,
            }
            if debug:
                result["full_response"] = (
                    full_response.model_dump()
                    if hasattr(full_response, "model_dump")
                    else full_response
                )
            return result

    def merge_result(result: Dict[str, Any]) -> None:
        qid = result["question_id"]
        record = questions_lookup[qid]
        record.setdefault("question_evaluation", {})
        ea: Dict[str, Any] = {
            "model": result["model"],
            "evaluation": result["evaluation"],
        }
        if debug and "full_response" in result:
            ea["full_response"] = result["full_response"]
        record["question_evaluation"]["evaluation_answering"] = ea
        processed_ids.add(qid)

    def sanitize_record_for_output(record: Dict[str, Any]) -> None:
        """Strip debug-only fields unless debug=true."""
        if debug:
            return
        ea = record.get("question_evaluation", {}).get("evaluation_answering")
        if isinstance(ea, dict):
            ea.pop("full_response", None)

    def snapshot_processed() -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for qid in input_order_ids:
            if qid in processed_ids:
                rec = questions_lookup.get(qid)
                if rec is not None:
                    out.append(rec)
        return out

    async def run() -> None:
        tasks = [asyncio.create_task(process_one(qid)) for qid in pending]
        items_since_last_save = 0
        async for coro in atqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Self-check eval",
        ):
            result = await coro
            if result is None:
                continue
            merge_result(result)
            items_since_last_save += 1
            if (
                periodic_save_interval > 0
                and items_since_last_save >= periodic_save_interval
            ):
                snapshot = snapshot_processed()
                for r in snapshot:
                    sanitize_record_for_output(r)
                save_to_json(snapshot, intermediate_json_path)
                items_since_last_save = 0
                logger.info(
                    f"Saved intermediate ({len(snapshot)} records) to "
                    f"{intermediate_json_path}"
                )

    if pending:
        asyncio.run(run())
    else:
        logger.info("Nothing to process — all records already evaluated.")

    # ---- Final save ----
    final: List[Dict[str, Any]] = []
    for qid in input_order_ids:
        rec = questions_lookup.get(qid)
        if rec and rec.get("question_evaluation", {}).get("evaluation_answering"):
            sanitize_record_for_output(rec)
            final.append(rec)

    save_to_json(final, output_json_path)
    logger.info(f"Saved full JSON ({len(final)} records) to {output_json_path}")

    # wo_context: deep copy with context_text removed
    final_wo_context: List[Dict[str, Any]] = []
    for rec in final:
        rec_copy = copy.deepcopy(rec)
        rec_copy.pop("context_text", None)
        final_wo_context.append(rec_copy)
    save_to_json(final_wo_context, output_json_wo_context_path)
    logger.info(
        f"Saved wo-context JSON ({len(final_wo_context)} records) to "
        f"{output_json_wo_context_path}"
    )

    # CSV: question_id, evaluation_answering_eval (in input order)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)) or ".", exist_ok=True)
    eval_by_id: Dict[str, Any] = {}
    for rec in final:
        eval_by_id[rec["question_id"]] = (
            rec.get("question_evaluation", {})
            .get("evaluation_answering", {})
            .get("evaluation")
        )
    import csv as _csv

    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(["question_id", "evaluation_answering_eval"])
        for qid in input_order_ids:
            if qid in eval_by_id:
                writer.writerow([qid, eval_by_id[qid]])
    logger.info(f"Saved CSV to {output_csv_path}")

    # Remove intermediate on clean completion
    if os.path.exists(intermediate_json_path):
        try:
            os.remove(intermediate_json_path)
            logger.info(f"Removed intermediate file {intermediate_json_path}")
        except OSError as e:
            logger.warning(
                f"Could not remove intermediate {intermediate_json_path}: {e}"
            )

    send_pushover_message(
        title="self-check-qa completed",
        message=(
            f"Evaluated {len(final)} / {len(input_order_ids)} records with "
            f"model={model}. Outputs: {output_json_path}, "
            f"{output_json_wo_context_path}, {output_csv_path}."
        ),
        priority=0,
        sound="magic",
        url="https://pushover.net/",
        url_title="Pushover Dashboard",
    )


if __name__ == "__main__":
    main()
