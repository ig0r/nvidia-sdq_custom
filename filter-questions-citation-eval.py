"""
Filter QA records produced by generate-qa.py based on citation quality.

Drops records whose full_citation.citation is missing/empty or matches one of
two known "no citation" sentinels:
  - code sentinel:   set by generate-qa.py:754-758 on retry exhaustion.
  - prompt sentinel: instructed by prompts/nemo_extract-citation.txt:37-44 as
                     the LLM's no-citation fallback.

Outputs (sibling of input):
  <input_stem>-c-eval.json          — kept records
  <input_stem>-c-eval-dropped.json  — dropped records with _drop_reason

See plans/srs-filter-questions-citation-eval.md for the full spec.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from pathlib import Path
from typing import Any, Optional

import loguru


logger = loguru.logger


REASON_CODE_SENTINEL = "code_sentinel"
REASON_PROMPT_SENTINEL = "prompt_sentinel"
REASON_MISSING_OR_EMPTY = "missing_or_empty"

DEFAULT_SENTINELS: list[str] = [
    "Max retries reached. No citation available.",
    "No relevant citation found",
]


# ---------------------------------------------------------------------------
# IO helpers (copied verbatim from generate-qa.py:132-146 to keep the script
# self-contained — same convention as its sibling).
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


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify(q: dict, sentinels: list[str]) -> Optional[str]:
    """Return None to keep, or one of the REASON_* strings to drop.

    Rules (first match wins):
      1. full_citation not a dict           -> missing_or_empty
      2. citation not a non-empty string    -> missing_or_empty
      3. citation == sentinels[0]           -> code_sentinel
      4. citation == sentinels[1]           -> prompt_sentinel
      5. otherwise                          -> None (keep)
    """
    fc = q.get("full_citation")
    if not isinstance(fc, dict):
        return REASON_MISSING_OR_EMPTY

    cit = fc.get("citation")
    if not isinstance(cit, str) or cit.strip() == "":
        return REASON_MISSING_OR_EMPTY

    if cit == sentinels[0]:
        return REASON_CODE_SENTINEL
    if cit == sentinels[1]:
        return REASON_PROMPT_SENTINEL

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _validate_sentinels(sentinels: Any) -> list[str]:
    if not isinstance(sentinels, list) or len(sentinels) != 2:
        logger.error(
            f"Config 'sentinels' must be a list of exactly 2 strings; got: {sentinels!r}"
        )
        sys.exit(2)
    for i, s in enumerate(sentinels):
        if not isinstance(s, str) or not s:
            logger.error(f"Config 'sentinels[{i}]' must be a non-empty string; got: {s!r}")
            sys.exit(2)
    return list(sentinels)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter QA records by citation quality (post-process generate-qa.py output)"
    )
    parser.add_argument(
        "--config",
        default="./filter-questions-citation-eval.toml",
        help="Path to TOML config",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console (stderr) log level. File log stays at DEBUG.",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level)
    os.makedirs("logs", exist_ok=True)
    logger.add(
        "logs/filter-questions-citation-eval.log",
        rotation="5 MB",
        compression="zip",
        level="DEBUG",
    )

    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(2)

    cfg_root = read_configuration(args.config)
    cfg = cfg_root.get("filter-questions-citation-eval")
    if not isinstance(cfg, dict):
        logger.error(
            f"Config table [filter-questions-citation-eval] missing or malformed in {args.config}"
        )
        sys.exit(2)

    input_file = cfg.get("input_file")
    if not isinstance(input_file, str) or not input_file:
        logger.error("Config key 'input_file' is required and must be a non-empty string")
        sys.exit(2)

    sentinels = _validate_sentinels(cfg.get("sentinels", DEFAULT_SENTINELS))

    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        sys.exit(2)

    try:
        records = load_json(input_file)
    except json.JSONDecodeError as e:
        logger.error(f"Input file is not valid JSON ({input_file}): {e}")
        sys.exit(2)

    if not isinstance(records, list):
        logger.error(
            f"Input file must contain a top-level JSON array; got {type(records).__name__}: {input_file}"
        )
        sys.exit(2)

    logger.info(f"Loaded {len(records)} records from {input_file}")
    if len(records) == 0:
        logger.warning("Input file is empty; writing empty kept and dropped files")

    kept: list[dict] = []
    dropped: list[dict] = []
    counts = {
        REASON_CODE_SENTINEL: 0,
        REASON_PROMPT_SENTINEL: 0,
        REASON_MISSING_OR_EMPTY: 0,
    }

    for q in records:
        if not isinstance(q, dict):
            q = {"_raw": q}
            q["_drop_reason"] = REASON_MISSING_OR_EMPTY
            counts[REASON_MISSING_OR_EMPTY] += 1
            dropped.append(q)
            continue

        reason = classify(q, sentinels)
        if reason is None:
            kept.append(q)
        else:
            q["_drop_reason"] = reason
            counts[reason] += 1
            dropped.append(q)

    input_path = Path(input_file)
    kept_path = str(input_path.with_name(f"{input_path.stem}-c-eval.json"))
    dropped_path = str(input_path.with_name(f"{input_path.stem}-c-eval-dropped.json"))

    save_to_json(kept, kept_path)
    save_to_json(dropped, dropped_path)

    total = len(records)
    n_kept = len(kept)
    n_dropped = len(dropped)
    kept_pct = (100.0 * n_kept / total) if total else 0.0

    if n_kept == 0 and total > 0:
        logger.warning("All records were dropped")

    logger.info(
        f"Summary: total={total}, kept={n_kept}, dropped={n_dropped}, kept_pct={kept_pct:.1f}%"
    )
    logger.info(f"  {REASON_CODE_SENTINEL}={counts[REASON_CODE_SENTINEL]}")
    logger.info(f"  {REASON_PROMPT_SENTINEL}={counts[REASON_PROMPT_SENTINEL]}")
    logger.info(f"  {REASON_MISSING_OR_EMPTY}={counts[REASON_MISSING_OR_EMPTY]}")
    logger.info(f"Wrote kept    -> {kept_path}")
    logger.info(f"Wrote dropped -> {dropped_path}")


if __name__ == "__main__":
    main()
