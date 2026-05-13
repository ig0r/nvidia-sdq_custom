# main module
# created 05.03.2026 00:29:31
import argparse
import json
import loguru
import sys
import os
import re
import pandas as pd
import asyncio
from typing import Dict, Any, Optional, Tuple, List

from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
from ollama import Client, AsyncClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import read_configuration, load_json, save_to_json
from inference import load_prompt_from_file, query_ollama_chat2
from response import QuestionAnswerEvaluationResponse

from utils import send_pushover_message

from dotenv import load_dotenv
load_dotenv()
import loguru

logger = loguru.logger
logger.add("logs/get-categorized-questions-eval-async2.log", rotation="5 MB", compression="zip", level="DEBUG")


async def query_ollama_chat2_async(
    client: AsyncClient,
    prompt: str,
    model: str,
    temperature: float = 0.0,
    system: Optional[str] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    format: Optional[Dict] = None,
    think: Optional[bool] = False
) -> Dict[str, Any]:
    """Async version of query_ollama_chat2 using AsyncClient."""
    try:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        options = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if top_p is not None:
            options["top_p"] = top_p
        if top_k is not None:
            options["top_k"] = top_k
        response = await client.chat(
            messages=messages,
            model=model,
            options=options,
            format=format,
            think=think
        )
        return response
    except Exception as e:
        error_message = str(e)
        if "connection" in error_message.lower():
            return {"error": f"Connection error to Ollama server: {error_message}"}
        else:
            return {"error": f"Error querying Ollama: {error_message}"}


def format_steps_as_numbered_list(steps: List[str]) -> str:
    """Format a list of steps as a numbered list string."""
    return "\n".join(f"{i + 1}. {step}" for i, step in enumerate(steps))

def generate_process_description(json_data: str) -> str:
    """Generate a formatted process description from a JSON process definition.
    Args:
        json_data: A JSON string containing the process definition with fields
                   such as name, description, source, steps, inputs, outputs,
                   and dependencies.
    Returns:
        A formatted string representing the process description.
    """
    data = json.loads(json_data)
    lines = []

    # Name, Description, Source
    lines.append(f"Name: {data['name']}")
    lines.append(f"Description: {data['description']}")
    lines.append(f"Source: {data['source']}")

    # Steps (1-indexed)
    lines.append("Steps:")
    for i, step in enumerate(data["steps"], start=1):
        lines.append(f"    {i}. {step}")

    # Inputs (1-indexed, includes source)
    lines.append("Inputs:")
    for i, inp in enumerate(data["inputs"], start=1):
        lines.append(f"    {i}. Name: {inp['name']}")
        lines.append(f"        Description: {inp['description']}")
        lines.append(f"        Source: {inp['source']}")

    # Outputs (1-indexed)
    lines.append("Outputs:")
    for i, out in enumerate(data["outputs"], start=1):
        lines.append(f"    {i}. Name: {out['name']}")
        lines.append(f"        Description: {out['description']}")

    # Dependencies (1-indexed)
    if data.get("dependencies"):
        lines.append("Dependencies:")
        for i, dep in enumerate(data["dependencies"], start=1):
            lines.append(f"    {i}. Name: {dep['name']}")
            lines.append(f"        Type: {dep['type']}")
            lines.append(f"        Description: {dep['description']}")
            lines.append(f"        Source: {dep['source']}")

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description='Evaluate categorized questions')
    parser.add_argument('--config', dest='config', type=str, default='get-categorized-questions-eval-async2.toml',
                        help='path to the config file in TOML format (default: get-categorized-questions-eval-async2.toml)')
    parser.add_argument('--port', dest='port', type=int, default=11434,
                        help='port number to connect to for the ollama server')
    parser.add_argument('--host', dest='host', type=str, default='localhost',
                        help='host name to connect to for the ollama server')
    args = parser.parse_args()

    config = read_configuration(args.config)

    # input with questions id to evaluate
    input_questions_ids = config["eval-categorized-questions"]["input_categorized_questions"]
    # input data with questions information
    input_questions_data = config["eval-categorized-questions"]["input_questions_data"]
    # input data with process information
    input_process_data = config["eval-categorized-questions"]["input_process_data"]
    # model evaluation
    model = config["eval-categorized-questions"]["model"]
    # prompt
    prompt_file_eval = config["eval-categorized-questions"]["prompt_eval"]
    # retries number for LLM queries
    retries_number = config["eval-categorized-questions"]["retries_number"]
    # when citation is missing, skip querying LLM and force evaluation value
    process_missining_citation = config["eval-categorized-questions"].get(
        "process_missining_citation",
        "No matching citation found",
    )
    # output json file
    output_json_file = config["eval-categorized-questions"]["output_json"]
    # output csv file
    output_csv_file = config["eval-categorized-questions"]["output_csv"]
    # intermediate results file for resuming
    intermediate_json_file = config["eval-categorized-questions"]["intermediate_json"]
    # periodic save interval
    periodic_save_interval = config["eval-categorized-questions"].get("periodic_save_interval", 10)
    # max concurrent questions for parallel processing
    max_concurrent_questions = config["eval-categorized-questions"].get("max_concurrent_questions", 1)
    # debug mode controls whether full_response is persisted
    debug = config["eval-categorized-questions"].get("debug", False)
    # phase flags
    evaluation_answering_enabled = config["eval-categorized-questions"].get("evaluation_answering", True)
    # phase-specific output files
    output_json_file_phase1 = config["eval-categorized-questions"]["output_json_phase1"]
    output_csv_file_phase1 = config["eval-categorized-questions"]["output_csv_phase1"]

    prompt_eval_template = load_prompt_from_file(prompt_file_eval)

    input_questions_ids_df = pd.read_csv(input_questions_ids)
    questions_data = load_json(input_questions_data)
    process_data = load_json(input_process_data)

    # Build lookup dicts for quick access
    questions_lookup = {q["question_id"]: q for q in questions_data}
    process_lookup = {p["id"]: p for p in process_data}

    client = Client(host=f"http://{args.host}:{args.port}")
    async_client = AsyncClient(host=f"http://{args.host}:{args.port}")

    def sanitize_question_for_output(question_record: Dict[str, Any]) -> None:
        """Remove internal fields that should not be persisted based on debug mode."""
        q_eval = question_record.get("question_evaluation", {})
        if isinstance(q_eval, dict):
            q_eval.pop("neural_network", None)
            q_eval.pop("eval_via_citation", None)
            if not debug:
                evaluation_answering = q_eval.get("evaluation_answering", {})
                if isinstance(evaluation_answering, dict):
                    evaluation_answering.pop("full_response", None)

        if not debug:
            # Source data may carry document citation either at the top level
            # or nested under a structured question payload.
            question_record.pop("document_citation", None)
            question_payload = question_record.get("question")
            if isinstance(question_payload, dict):
                question_payload.pop("document_citation", None)

    # =========== PHASE 1: Evaluation Answering ===========
    if evaluation_answering_enabled:
        logger.info("Starting Phase 1: Evaluation Answering")

        # Load intermediate results for phase 1 if they exist
        phase1_processed_ids = set()
        if os.path.exists(intermediate_json_file):
            try:
                with open(intermediate_json_file, 'r', encoding='utf-8') as f:
                    phase1_intermediate = json.load(f)
                phase1_processed_ids = {q["question_id"] for q in phase1_intermediate}
                # Merge loaded results into questions_lookup
                for q in phase1_intermediate:
                    qid = q["question_id"]
                    if qid in questions_lookup:
                        questions_lookup[qid].setdefault("question_evaluation", {})
                        ea = q.get("question_evaluation", {}).get("evaluation_answering")
                        if ea:
                            questions_lookup[qid]["question_evaluation"]["evaluation_answering"] = ea
                logger.info(f"Loaded {len(phase1_intermediate)} phase 1 intermediate results from {intermediate_json_file}, resuming...")
            except Exception as e:
                logger.warning(f"Failed to load phase 1 intermediate results from {intermediate_json_file}: {e}. Starting fresh.")
                phase1_processed_ids = set()
        elif os.path.exists(output_json_file_phase1):
            try:
                with open(output_json_file_phase1, 'r', encoding='utf-8') as f:
                    phase1_output = json.load(f)
                phase1_processed_ids = {q["question_id"] for q in phase1_output}
                for q in phase1_output:
                    qid = q["question_id"]
                    if qid in questions_lookup:
                        questions_lookup[qid].setdefault("question_evaluation", {})
                        ea = q.get("question_evaluation", {}).get("evaluation_answering")
                        if ea:
                            questions_lookup[qid]["question_evaluation"]["evaluation_answering"] = ea
                logger.info(f"No intermediate file but found phase 1 output file {output_json_file_phase1}, loaded {len(phase1_output)} results. Skipping already processed.")
            except Exception as e:
                logger.warning(f"Failed to load phase 1 output from {output_json_file_phase1}: {e}. Starting fresh.")
                phase1_processed_ids = set()
        else:
            logger.info("No phase 1 intermediate results found, starting fresh")

        # Build list of questions to process (preserving order)
        phase1_tasks = []
        for _, row in input_questions_ids_df.iterrows():
            question_id = row["question_id"]
            if question_id in phase1_processed_ids:
                continue
            question_info = questions_lookup.get(question_id)
            if question_info is None:
                logger.warning(f"Question ID '{question_id}' not found in questions data")
                continue
            process_info = process_lookup.get(question_info["process_id"])
            if process_info is None:
                logger.warning(f"Process ID '{question_info['process_id']}' not found in process data")
                continue
            phase1_tasks.append((question_id, question_info, process_info))

        semaphore = asyncio.Semaphore(max_concurrent_questions)

        async def phase1_process_question(question_id, question_info, process_info):
            """Process a single question for Phase 1 (async with semaphore)."""
            async with semaphore:
                question_text = question_info["question"]
                answer_text = question_info.get("answer", question_info.get("anser", ""))
                # element_type = question_info["element"]["type"]
                # element_index = question_info["element"]["index"]

                # logger.debug(f"Phase 1 - Processing question_id: {question_id}")
                # logger.debug(f"  Question: {question_text}")
                # logger.debug(f"  Process ID: {question_info['process_id']}")
                # logger.debug(f"  Element type: {element_type}, Element index: {element_index}")

                process_citation = process_info.get("citation", {}).get("citation", "")
                if process_citation.strip() == process_missining_citation:
                    logger.debug(
                        f"Skipping LLM for question_id={question_id}: process citation marked as missing."
                    )
                    return {
                        "question_id": question_id,
                        "model": model,
                        "evaluation": -1,
                    }

                prompt_query = (
                    prompt_eval_template
                    .replace("{QUESTION}", question_text)
                    .replace("{ANSWER}", answer_text)
                    .replace("{PROCESS_CITATION}", process_citation)
                )
                # logger.debug(f"  Constructed prompt:\n{prompt_query}")

                # if element_type in ("general", "input", "output"):
                #     steps_text = format_steps_as_numbered_list(process_info["steps"])
                #     prompt_query = prompt_general_eval_template.replace("{QUESTION}", question_text).replace("{STEPS}", steps_text)
                #     logger.debug(f"  Using general eval prompt for element_type: {element_type}")
                # elif element_type == "step":
                #     step_text = process_info["steps"][element_index]
                #     prompt_query = prompt_step_eval_template.replace("{QUESTION}", question_text).replace("{ELEMENT}", step_text)
                #     logger.debug(f"  Using step eval prompt, step: {step_text}")
                # else:
                #     logger.warning(f"  Unknown element type: {element_type}")
                #     return None

                # logger.debug(f"  Constructed prompt:\n{prompt_query}")

                llm_retry = 0
                llm_result = None
                full_response = None
                while llm_retry < retries_number:
                    try:
                        response = await query_ollama_chat2_async(
                            async_client,
                            prompt_query,
                            model=model,
                            temperature=0.0,
                            format=QuestionAnswerEvaluationResponse.model_json_schema(),
                            think=False
                        )
                        response_text = response.get('message', {}).get('content', '')
                        #logger.debug(f"response:\n{response}")
                        llm_result = QuestionAnswerEvaluationResponse.model_validate_json(response_text)
                        # logger.debug(f"Question {question_id}: {question_text} -> LLM evaluation result: {llm_result.evaluation}.")
                        full_response = response
                        # logger.debug(f"Full LLM response for question_id={question_id}:\n{full_response}")
                        break
                    except Exception as e:
                        llm_retry += 1
                        logger.warning(f"  Retry {llm_retry}/{retries_number} for question_id={question_id}: {e}")

                if llm_result is None:
                    logger.error(f"  Failed to evaluate question_id={question_id} after {retries_number} retries")
                    return None

                # logger.info(f"  Evaluated question_id={question_id}: evaluation={llm_result.evaluation}")
                result = {
                    "question_id": question_id,
                    "model": model,
                    "evaluation": llm_result.evaluation,
                }
                if debug:
                    result["full_response"] = full_response.model_dump() if hasattr(full_response, 'model_dump') else full_response
                return result

        async def run_phase1():
            tasks = [phase1_process_question(qid, qinfo, pinfo) for qid, qinfo, pinfo in phase1_tasks]
            results = []
            items_since_last_save = 0
            for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Phase 1: Evaluating questions"):
                result = await coro
                results.append(result)
                if result is not None:
                    qid = result["question_id"]
                    question_info = questions_lookup[qid]
                    question_info["question_evaluation"] = question_info.get("question_evaluation", {})
                    evaluation_answering_data = {
                        "model": result["model"],
                        "evaluation": result["evaluation"],
                    }
                    if debug and "full_response" in result:
                        evaluation_answering_data["full_response"] = result["full_response"]
                    question_info["question_evaluation"]["evaluation_answering"] = evaluation_answering_data
                    phase1_processed_ids.add(qid)
                    items_since_last_save += 1

                    if periodic_save_interval > 0 and items_since_last_save >= periodic_save_interval:
                        phase1_save = [questions_lookup[qid] for qid in phase1_processed_ids if qid in questions_lookup]
                        for q in phase1_save:
                            sanitize_question_for_output(q)
                        save_to_json(phase1_save, intermediate_json_file)
                        items_since_last_save = 0
                        logger.info(f"Saved phase 1 intermediate results ({len(phase1_save)} items) to {intermediate_json_file}")

        asyncio.run(run_phase1())

        # Clean up phase 1 intermediate file after successful completion
        if os.path.exists(intermediate_json_file):
            os.remove(intermediate_json_file)
            logger.info(f"Removed phase 1 intermediate file {intermediate_json_file}")

        logger.info("Phase 1: Evaluation Answering completed")

        # Save Phase 1 outputs
        phase1_results = []
        for _, row in input_questions_ids_df.iterrows():
            qid = row["question_id"]
            q = questions_lookup.get(qid)
            if q and q.get("question_evaluation", {}).get("evaluation_answering"):
                phase1_results.append(q)

        for q in phase1_results:
            sanitize_question_for_output(q)

        save_to_json(phase1_results, output_json_file_phase1)
        logger.info(f"Saved Phase 1 JSON results to {output_json_file_phase1}")

        phase1_csv_df = input_questions_ids_df.copy()
        phase1_eval_map = {}
        for q in phase1_results:
            qid = q["question_id"]
            ea = q.get("question_evaluation", {}).get("evaluation_answering", {})
            phase1_eval_map[qid] = {
                "evaluation_answering_eval": ea.get("evaluation"),
            }
        for col in ["evaluation_answering_eval"]:
            phase1_csv_df[col] = phase1_csv_df["question_id"].map(
                lambda qid, c=col: phase1_eval_map.get(qid, {}).get(c)
            )
        drop_cols_phase1 = [
            c for c in ["neural_network", "eval_via_citation", "eval_via_citation_eval", "eval"]
            if c in phase1_csv_df.columns
        ]
        if drop_cols_phase1:
            phase1_csv_df = phase1_csv_df.drop(columns=drop_cols_phase1)

        phase1_csv_df.to_csv(output_csv_file_phase1, index=False)
        logger.info(f"Saved Phase 1 CSV results to {output_csv_file_phase1}")

    # =========== Final Save ===========
    # Build final results: all questions that have any evaluation data
    final_results = []
    for _, row in input_questions_ids_df.iterrows():
        qid = row["question_id"]
        q = questions_lookup.get(qid)
        if q and q.get("question_evaluation"):
            final_results.append(q)

    # Ensure unsupported metadata is not persisted in JSON output.
    for q in final_results:
        sanitize_question_for_output(q)

    save_to_json(final_results, output_json_file)
    logger.info(f"Saved evaluation results to {output_json_file}")

    # Save output CSV
    eval_map = {}
    for q in final_results:
        qid = q["question_id"]
        ea = q.get("question_evaluation", {}).get("evaluation_answering", {})
        eval_map[qid] = {
            "evaluation_answering_eval": ea.get("evaluation"),
        }

    for col in ["evaluation_answering_eval"]:
        input_questions_ids_df[col] = input_questions_ids_df["question_id"].map(
            lambda qid, c=col: eval_map.get(qid, {}).get(c)
        )

    drop_cols_final = [
        c for c in ["neural_network", "eval_via_citation", "eval_via_citation_eval", "eval"]
        if c in input_questions_ids_df.columns
    ]
    if drop_cols_final:
        input_questions_ids_df = input_questions_ids_df.drop(columns=drop_cols_final)

    input_questions_ids_df.to_csv(output_csv_file, index=False)
    logger.info(f"Saved evaluation CSV to {output_csv_file}")

    send_pushover_message(
        title=f"get-process-self-consistency-async completed",
        message=f"Checked {len(input_questions_ids_df)} elements (inputs and outputs). Saved results in {output_json_file} and {output_csv_file} 🚀",
        priority=0,
        sound="magic",
        url="https://pushover.net/",
        url_title="Pushover Dashboard"
    )


if __name__ == "__main__":
    main()

