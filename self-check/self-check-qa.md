# `self-check-qa.py`

LLM-as-judge **self-check** for the QA records produced by
[`generate-qa.py`](../generate-qa.py). For every record in a single input JSON,
the script sends `(question, answer, context_text)` to an Ollama-hosted LLM
with a strict schema-bound prompt and records whether the model believes the
answer is supported by the context (`1` = Yes, `0.5` = N/A, `0` = No,
`-1` = skipped because the record had no `context_text`).

The script is fully **asynchronous**, **resumable**, **self-contained**
(no imports from sibling projects), and intended to run against a local Ollama
server — locally on `gpt-oss:20b` for debugging, on the cluster on
`gpt-oss:120b` for production runs.

It is an adaptation of [`example/get-categorized-questions-eval-async2.py`](example/get-categorized-questions-eval-async2.md);
see that doc for the original design rationale.

---

## 1. What it does

```
for each record in input_qa_json:
    question     = record["question"]
    answer       = record["answer"]
    context_text = record["context_text"]

    if not context_text.strip():
        evaluation = -1                              # skip — no context
    else:
        prompt = prompt_template
                   .replace("{QUESTION}",         question)
                   .replace("{ANSWER}",           answer)
                   .replace("{PROCESS_CITATION}", context_text)
        response   = await ollama.chat(..., format=QuestionAnswerEvaluationResponse.model_json_schema())
        evaluation = response.evaluation             # ∈ {0, 0.5, 1}

    record["question_evaluation"]["evaluation_answering"] = {
        "model": model,
        "evaluation": evaluation,
    }
```

The LLM is forced into the JSON shape declared by
`QuestionAnswerEvaluationResponse` (defined inline in the script) via Ollama's
`format=` parameter, so the script only has to parse one float field. Values
outside `{0, 0.5, 1}` are rejected by Pydantic's validator and trigger a retry.

> **Why "self-check"?** The QA pairs were generated upstream from artifacts
> extracted from the same `context_text`. Re-presenting only the raw context
> and asking the LLM whether the answer is supported is a one-shot consistency
> check: it surfaces hallucinations and answers that drifted from the source.

---

## 2. Files

```
self-check/
├── self-check-qa.py                # the script
├── self-check-qa.toml              # config
├── self-check-qa.md                # this doc
├── prompts/
│   └── evaluate-process-question-answer-02.txt   # the judge prompt
└── self-check-output/              # outputs (created at runtime)
    ├── self-check-qa-results.json             (full records + evaluation)
    ├── self-check-qa-results_wo_context.json  (same, with context_text stripped)
    ├── self-check-qa-results.csv              (flat: question_id, evaluation_answering_eval)
    └── self-check-qa-intermediate.json        (rolling checkpoint; removed on clean completion)
```

The prompt file is a copy of `self-check/example/prompts/evaluate-process-question-answer-02.txt`;
the new script has its own `prompts/` directory so it is independent of the
example.

---

## 3. Input

`input_qa_json` — a single JSON list of records as produced by
[`generate-qa.py`](../generate-qa.md). Each record must contain at least:

| field          | used as                                  |
| -------------- | ---------------------------------------- |
| `question_id`  | identity key + output ordering           |
| `question`     | `{QUESTION}` in the prompt               |
| `answer`       | `{ANSWER}` in the prompt                 |
| `context_text` | `{PROCESS_CITATION}` in the prompt       |

Any other fields (`full_citation`, `question_type`, `doc_id`, `u_ctx_id`,
`artifact`, `model_qa`, …) are preserved verbatim in the JSON output.

If `context_text` is missing or empty after `str.strip()`, the LLM is **not**
called and the record is hard-set to `evaluation = -1`. This mirrors the
example's "missing citation" sentinel handling, adapted to the new schema.

---

## 4. Prompt — `prompts/evaluate-process-question-answer-02.txt`

```
Context: {PROCESS_CITATION}
Question: {QUESTION}
Answer: {ANSWER}
Is Answer supported by the provided Context with respect to the Question?
Answer Yes, No, or N/A the following JSON format:
{{
"evaluation": <1 if Yes, 0 if No, or 0.5 if N/A>,
}}
```

Substitution is plain `str.replace`, not a template engine — keep the
placeholder casing exact (`{QUESTION}`, `{ANSWER}`, `{PROCESS_CITATION}`).
The doubled braces around the JSON example survive `replace` untouched, so
the LLM still sees single braces.

The JSON shape is also enforced server-side via Ollama's `format=` argument
(`QuestionAnswerEvaluationResponse.model_json_schema()`); the prompt's JSON
sketch is a redundant guard-rail. The relevant contract is the Pydantic model:

```python
class QuestionAnswerEvaluationResponse(BaseModel):
    evaluation: float   # must be 0, 0.5, or 1
```

---

## 5. Configuration (`self-check-qa.toml`)

All knobs live under the `[self-check-qa]` table:

| key                          | type | what it controls                                                                 |
| ---------------------------- | ---- | -------------------------------------------------------------------------------- |
| `input_qa_json`              | path | input JSON list (output of `generate-qa.py`)                                     |
| `prompt_eval`                | path | path to the prompt template                                                      |
| `model`                      | str  | Ollama model tag (e.g. `gpt-oss:120b` for cluster, `gpt-oss:20b` for local)      |
| `output_json`                | path | full records + evaluation                                                        |
| `output_json_wo_context`     | path | same records with `context_text` stripped (saves space)                          |
| `output_csv`                 | path | flat CSV (`question_id, evaluation_answering_eval`) in input order               |
| `intermediate_json`          | path | rolling save file used to **resume interrupted runs**                            |
| `retries_number`             | int  | attempts per record before giving up                                             |
| `periodic_save_interval`     | int  | flush `intermediate_json` every N completed records (`0` disables)               |
| `max_concurrent_questions`   | int  | `asyncio.Semaphore` bound — number of in-flight LLM requests                     |
| `debug`                      | bool | when `true`, persists `full_response` in the JSON output (large)                 |

> **Concurrency vs. Ollama**: `max_concurrent_questions` should be tuned
> against the Ollama server's parallelism budget. The example SLURM job for
> the original script uses `max_concurrent_questions = 50` against
> `OLLAMA_NUM_PARALLEL=8` / `OLLAMA_MAX_QUEUE=512` — the same shape applies
> here. For local `gpt-oss:20b` debugging, drop it to `2`–`4`.

---

## 6. Running locally

Prerequisites:
1. An Ollama server reachable at `--host:--port` (default `localhost:11434`)
   with `model` already pulled (`ollama pull gpt-oss:20b`).
2. The project virtualenv at `.venv/` (or an env with `loguru`, `ollama`,
   `pydantic`, `tqdm`, `python-dotenv`, optional `requests` for Pushover).

```bash
.venv/bin/python self-check/self-check-qa.py \
    --config self-check/self-check-qa.toml \
    --host   localhost \
    --port   11434
```

The script logs to `logs/self-check-qa.log` (rotated at 5 MB, zipped). The
`logs/` directory is auto-created on startup, so no manual setup is needed.

---

## 7. Running on the cluster

Set up Ollama on a compute node with `gpt-oss:120b` pulled, then run:

```bash
.venv/bin/python self-check/self-check-qa.py \
    --config self-check/self-check-qa.toml \
    --host   localhost \
    --port   $OLLAMA_PORT
```

with `model = "gpt-oss:120b"` in the config. The same SLURM-side env-var
tuning recommended for the example script applies here — see
[`example/get-categorized-questions-eval-async2.md` §7](example/get-categorized-questions-eval-async2.md#7-running-on-slurm)
for the full table (`OLLAMA_CONTEXT_LENGTH=65536`, `OLLAMA_NUM_PARALLEL=8`,
`OLLAMA_MAX_QUEUE=512`, `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_NO_CLOUD=1`).

---

## 8. Resumption

Two on-disk artifacts make the script resumable:

1. **`intermediate_json`** — written every `periodic_save_interval` completions
   *during* a run. On startup, if this file exists, the script loads the
   `question_id`s it contains and skips them. The file is **deleted** once
   the run finishes cleanly.
2. **`output_json`** — written *after* the run finishes. If
   `intermediate_json` is absent on startup but `output_json` exists, the
   script treats it as a completed checkpoint and skips its `question_id`s.

Three useful restart modes:

| state on disk                              | behavior                               |
| ------------------------------------------ | -------------------------------------- |
| neither present                            | full run from scratch                  |
| `intermediate_json` only                   | mid-run crash — resume from last flush |
| `output_json` only (no intermediate)       | run already complete — re-runs are no-ops |

**To force a clean re-run**, delete both files (and `output_json_wo_context`
and `output_csv` if you also want those rewritten from a fresh evaluation).
There is no CLI flag for this and no `overwrite` config key.

---

## 9. Outputs

### 9.1 `output_json` — full records + evaluation

Each input record passed through verbatim with one block added:

```json
{
  "question_id": "TBF000001_UKN000-ctx-0-q-0",
  "question":    "What is the main focus of this study?",
  "answer":      "This study primarily focuses on …",
  "context_text": "MORE INFORMATION [www.pcccenter.iastate.edu] …",
  "full_citation": { "citation": "...", "first_sentence": "...", "last_sentence": "..." },
  "question_type":  "factual",
  "...": "all other fields from generated-questions.json passed through",
  "question_evaluation": {
    "evaluation_answering": {
      "model":      "gpt-oss:120b",
      "evaluation": 1.0
    }
  }
}
```

Records that failed every retry are **omitted** from the output. Records
skipped because `context_text` was empty appear with `evaluation = -1`.

When `debug = true`, `evaluation_answering` also carries `full_response` —
the raw Ollama response dict.

### 9.2 `output_json_wo_context` — same data, smaller

Identical to `output_json` but with `context_text` removed from every record.
For a ~41 k-record corpus, this typically reduces file size by ~40 %.

### 9.3 `output_csv` — flat per-record evaluation

```csv
question_id,evaluation_answering_eval
TBF000001_UKN000-ctx-0-q-0,1.0
TBF000001_UKN000-ctx-0-q-1,1.0
...
```

Row order matches the input JSON order (which `generate-qa.py` emits
sorted/renumbered). Valid values in `evaluation_answering_eval`:

| value  | meaning                                                                  |
| ------ | ------------------------------------------------------------------------ |
| `1.0`  | LLM judged the answer **is** supported by `context_text`                 |
| `0.5`  | LLM judged it **partial / N/A**                                          |
| `0.0`  | LLM judged the answer is **not** supported                               |
| `-1`   | Skipped — record had no `context_text`                                   |

Records that failed every retry are not present in the CSV.

---

## 10. Architecture and control flow

```
main()
 ├─ read_configuration(args.config)
 ├─ load_json(input_qa_json)               → list[dict]
 ├─ AsyncClient(host=…)
 │
 ├─ try to resume:
 │     intermediate_json → else output_json → else fresh
 │
 ├─ build pending = [qid for qid in input_order if not processed]
 ├─ semaphore = asyncio.Semaphore(max_concurrent_questions)
 │
 ├─ asyncio.run(run):
 │     tasks = [process_one(qid) for qid in pending]
 │     async for done in atqdm(asyncio.as_completed(tasks)):
 │         result = await done
 │         questions_lookup[qid]["question_evaluation"]["evaluation_answering"] = ...
 │         every N items → save_to_json(snapshot, intermediate_json)
 │
 ├─ save full JSON  (input order)
 ├─ save wo_context JSON  (input order, context_text removed)
 ├─ save CSV  (input order)
 ├─ remove intermediate_json   (clean shutdown)
 └─ send_pushover_message(...)
```

The hot path inside `process_one`:

```
async with semaphore:
    if not context_text.strip():
        return {evaluation: -1}
    prompt = template.replace(…)
    for attempt in 1..retries_number:
        resp = await async_client.chat(model=model, format=schema, temperature=0.0, think=False)
        result = QuestionAnswerEvaluationResponse.model_validate_json(resp.message.content)
        break    # on success
    return {question_id, model, evaluation: result.evaluation}
```

`temperature=0.0` and `think=False` are hard-coded — the run is intended to
be deterministic and to bypass `gpt-oss`-style chain-of-thought emission so
the response body is just the schema-bound JSON.

---

## 11. Notes and gotchas

- **Input order is authoritative.** Output JSON and CSV row order mirror the
  input JSON, *not* completion order. The async runner uses
  `asyncio.as_completed` for live progress, but the final outputs are
  reconstructed by walking `questions_lookup` in input order.
- **Failed records are dropped.** A record that exhausted its retries is not
  written to `output_json` / `output_csv`. Failures are logged at ERROR. If
  you need every record represented, raise `retries_number` or check the log.
- **`debug = true` blows up the JSON.** `full_response` adds kilobytes per
  record; for a 40 k-record corpus that's hundreds of MB. Use only for
  spot-checks, not full runs.
- **Pushover is best-effort.** `send_pushover_message` POSTs to Pushover if
  `PUSHOVER_USER` and `PUSHOVER_TOKEN` are set; otherwise it logs a warning
  and no-ops. No `requests`? Same — warning and no-op.
- **Different corpora ⇒ different config files.** All paths come from the
  TOML. Keep one TOML per dataset (e.g. `self-check-qa-briefs.toml`,
  `self-check-qa-techbriefs.toml`) and pass them via `--config`.
- **No CLI flag to override `input_qa_json` / outputs.** Same reason —
  paths are TOML-only. Edit the config or copy it.
