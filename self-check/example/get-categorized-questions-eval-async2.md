# `get-categorized-questions-eval-async2.py`

LLM-as-judge **self-check** for a corpus of generated `(question, answer)` pairs.
For every question id in a CSV, the script looks up the corresponding QA pair and
the citation of the *process* the question was generated from, sends them to an
Ollama-hosted LLM with a strict schema-bound prompt, and records whether the
model believes the answer is supported by the citation (`1` = Yes, `0.5` = N/A,
`0` = No, `-1` = skipped because the source process had no citation).

The script is fully **asynchronous**, **resumable**, and intended to run against
a local Ollama server (e.g. spun up on a SLURM compute node — see
[`cluster/questions-eval-pub242.slurm`](cluster/questions-eval-pub242.slurm)).

---

## 1. What it does

The pipeline implements one logical phase — **Phase 1: Evaluation Answering** —
which performs a *single* LLM judgment per question:

```
for each question_id in pub242_question_ids.csv:
    question      = questions_lookup[question_id].question
    answer        = questions_lookup[question_id].answer
    process_id    = questions_lookup[question_id].process_id
    citation_text = process_lookup[process_id].citation.citation

    if citation_text == "No matching citation found":
        evaluation = -1                                # skip — no ground truth
    else:
        prompt    = prompt_template
                      .replace("{QUESTION}",         question)
                      .replace("{ANSWER}",           answer)
                      .replace("{PROCESS_CITATION}", citation_text)
        response  = await ollama.chat(..., format=QuestionAnswerEvaluationResponse.model_json_schema())
        evaluation = response.evaluation              # ∈ {0, 0.5, 1}
```

The LLM is *forced* into the JSON shape declared by
`QuestionAnswerEvaluationResponse` (a Pydantic model in
`examples/response.py`) — Ollama's `format=` parameter constrains the decoder
to that schema, so the script only has to parse one float field. Values
outside `{0, 0.5, 1}` are rejected by Pydantic's validator and trigger a retry.

> **Why "self-check"?** The QA pairs were generated upstream from process descriptions.
> Re-presenting only the *raw citation* (not the structured process description
> that drove generation) and asking the LLM whether the answer is supported is
> a one-shot consistency check: it surfaces hallucinations and citations that
> drifted from the source text.

The script also keeps a stale-removed sanitizer
(`sanitize_question_for_output`) that strips internal fields
(`neural_network`, `eval_via_citation`, `document_citation`, and `full_response`
when `debug = false`) before persisting.

---

## 2. Files in this example

```
self-check/example/
├── get-categorized-questions-eval-async2.py            # the script
├── get-categorized-questions-eval-async2-pub242.toml   # config for the pub242 run
├── prompts/
│   └── evaluate-process-question-answer-02.txt         # the judge prompt
├── cluster/
│   └── questions-eval-pub242.slurm                     # SLURM submission script
└── processed-questions-pub242-chapters/                # inputs + outputs for the run
    ├── pub242_question_ids.csv                                 (input)  ~15 k ids
    ├── generated-questions_5_async_citation_prompt_complete.json (input) QA records
    ├── process-citations_v5.json                                (input) process records + citations
    └── output-questions-eval2/
        ├── categorized-questions-eval-results-phase1-pub242.json   (output)
        ├── categorized-questions-eval-results-phase1-pub242.csv    (output)
        ├── categorized-questions-eval-results-pub242.json          (output, identical to phase1 today)
        └── categorized-questions-eval-results-pub242.csv           (output, identical to phase1 today)
```

The script's runtime imports (`utils`, `inference`, `response`) live in
`examples/` at the repo root and are picked up via a `sys.path.append("..")`
that assumes the script is laid out beside them in the working tree of the
machine that runs it (this example folder is a snapshot for documentation; on
the cluster the layout flattens those modules next to the script — see
[§7](#7-running-on-slurm)).

---

## 3. Inputs

### 3.1 `input_categorized_questions` — CSV of question ids to evaluate

Single column `question_id`. Acts as the *driver* — only ids listed here are
processed, and the output CSV preserves the same row order.

```csv
question_id
publication_242-publication_242_chapter_6_section_6.1-proc-0-q-0
publication_242-publication_242_chapter_6_section_6.1-proc-0-q-1
...
```

### 3.2 `input_questions_data` — JSON of QA records (lookup by `question_id`)

Each element must contain at least:

| field           | used as                                       |
| --------------- | --------------------------------------------- |
| `question_id`   | join key                                      |
| `question`      | `{QUESTION}` in the prompt                    |
| `answer`        | `{ANSWER}` in the prompt (falls back to `anser` for a known upstream typo) |
| `process_id`    | join key into `input_process_data`            |

Any other fields (`citation`, `element`, `question_type`, …) are preserved
verbatim in the JSON output but otherwise unused by this script.

### 3.3 `input_process_data` — JSON of process records (lookup by `id`)

Each element must contain at least:

| field                   | used as                                    |
| ----------------------- | ------------------------------------------ |
| `id`                    | join key (== `question.process_id`)        |
| `citation.citation`     | `{PROCESS_CITATION}` in the prompt         |

If `citation.citation` equals the configured `process_missining_citation`
marker (default: `"No matching citation found"`), the LLM call is skipped and
the evaluation is hard-set to `-1`.

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

Placeholder substitution is done by plain `str.replace`, not a template
engine — keep the placeholder casing exact (`{QUESTION}`, `{ANSWER}`,
`{PROCESS_CITATION}`). The doubled braces around the JSON example are an
artifact of f-string-friendly authoring; the LLM still sees single braces
because they pass through `replace` untouched.

The JSON shape is *also* enforced server-side by passing
`QuestionAnswerEvaluationResponse.model_json_schema()` as Ollama's `format=`
argument, so the prompt's JSON sketch is mostly redundant guard-rail and the
relevant contract is the Pydantic model:

```python
class QuestionAnswerEvaluationResponse(BaseModel):
    evaluation: float   # must be 0, 0.5, or 1
```

---

## 5. Configuration (`*.toml`)

All knobs live under the `[eval-categorized-questions]` table. Example
(`get-categorized-questions-eval-async2-pub242.toml`):

| key                          | type   | what it controls                                                                 |
| ---------------------------- | ------ | -------------------------------------------------------------------------------- |
| `input_categorized_questions`| path   | CSV of `question_id`s to evaluate (driver — see [§3.1](#31-input_categorized_questions--csv-of-question-ids-to-evaluate)) |
| `input_questions_data`       | path   | JSON of QA records keyed by `question_id`                                        |
| `input_process_data`         | path   | JSON of process records keyed by `id`                                            |
| `model`                      | str    | Ollama model tag (e.g. `gpt-oss:120b`, `ministral-3:14b-instruct-2512-q8_0`)     |
| `prompt_eval`                | path   | path to the prompt template                                                      |
| `retries_number`             | int    | attempts per question before giving up and writing `None`                        |
| `process_missining_citation` | str    | sentinel string — when found in `citation.citation`, evaluation is forced to `-1` |
| `output_json`                | path   | final per-question JSON                                                          |
| `output_csv`                 | path   | final flat CSV (`question_id, evaluation_answering_eval`)                        |
| `output_json_phase1`         | path   | Phase 1 JSON (currently same content as `output_json`)                           |
| `output_csv_phase1`          | path   | Phase 1 CSV                                                                      |
| `intermediate_json`          | path   | rolling save file used to **resume interrupted runs**                            |
| `periodic_save_interval`     | int    | flush `intermediate_json` every N completed questions (`0` disables)             |
| `max_concurrent_questions`   | int    | `asyncio.Semaphore` bound — number of in-flight LLM requests                     |
| `debug`                      | bool   | when `true`, persists `full_response` in the JSON output                         |
| `evaluation_answering`       | bool   | gate for Phase 1; if `false`, the script falls straight through to the Final Save and emits empty CSV columns |

> **Note — concurrency vs. Ollama**: `max_concurrent_questions` must be tuned
> against the Ollama server's parallelism budget. On the example SLURM job we
> use `max_concurrent_questions = 50` against an Ollama server started with
> `OLLAMA_NUM_PARALLEL=8` and `OLLAMA_MAX_QUEUE=512`, so up to 50 requests
> queue at the client and Ollama services 8 of them in parallel on one GPU.

---

## 6. Running locally

Prerequisites:
1. An Ollama server reachable at `--host:--port` (default `localhost:11434`)
   with the model named in `model` already pulled.
2. A `.env` file in CWD if any code path needs API keys; this script itself
   only talks to Ollama, but the imports load `dotenv` eagerly.
3. The `examples/` modules (`utils`, `inference`, `response`) reachable on
   `PYTHONPATH` — on the cluster they sit next to the script; locally you can
   either copy them in or add `examples/` to `PYTHONPATH`.

```bash
python get-categorized-questions-eval-async2.py \
    --config get-categorized-questions-eval-async2-pub242.toml \
    --host   localhost \
    --port   11434
```

The script logs to `logs/get-categorized-questions-eval-async2.log` (rotated
at 5 MB, zipped). The `logs/` directory must exist — create it before the
first run.

---

## 7. Running on SLURM

`cluster/questions-eval-pub242.slurm` is the production job. The job

1. Allocates one H200 GPU on the `gpu` cluster for up to ~2 days.
2. Picks a free TCP port on the compute node and binds Ollama to it via
   `OLLAMA_HOST`.
3. Starts `ollama serve` in the background with tuned env vars:

   | env var                 | value   | effect                                    |
   | ----------------------- | ------- | ----------------------------------------- |
   | `OLLAMA_CONTEXT_LENGTH` | `65536` | needed because some citations are long    |
   | `OLLAMA_NUM_PARALLEL`   | `8`     | concurrent decode streams per model       |
   | `OLLAMA_MAX_QUEUE`      | `512`   | request backlog before 429s                |
   | `OLLAMA_FLASH_ATTENTION`| `1`     | FA2 kernels (faster decode, lower VRAM)   |
   | `OLLAMA_NO_CLOUD`       | `1`     | force local-only model resolution         |

4. Activates the project conda env (`/ix1/lkhazanovich/igs18/envs/proc-env`)
   and runs the script against `localhost:$PORT`.

The job pins both `ollama serve` and the Python client on the same node, so
`--host localhost` is correct — there's no cross-node networking. The example
on disk uses `gpt-oss:120b`; the commented alternative
`ministral-3:14b-instruct-2512-q8_0` is what was used for the earlier
ministral-eval run sitting alongside the gpt-oss-eval outputs in
`output-questions-eval2/`.

---

## 8. Resumption and caching

Two distinct on-disk artifacts make the script resumable:

1. **`intermediate_json`** — written every `periodic_save_interval` completions
   *during* a run. On startup, if this file exists, the script loads the
   `question_id`s it contains and skips them. The file is **deleted** once
   Phase 1 finishes cleanly.
2. **`output_json_phase1`** — written *after* Phase 1 finishes. If
   `intermediate_json` is absent on startup but `output_json_phase1` exists,
   the script treats it as a completed-checkpoint and resumes from there.

This gives three useful restart modes:

| state on disk                                  | behavior                                |
| ---------------------------------------------- | --------------------------------------- |
| neither present                                | full run from scratch                   |
| `intermediate_json` only                       | mid-run crash — resume from last flush  |
| `output_json_phase1` only (no intermediate)    | Phase 1 already done — re-runs are no-ops |

**To force a clean re-run**, delete *both* files. There is no CLI flag for
this and no `overwrite` config key.

---

## 9. Outputs

### 9.1 JSON (`output_json` / `output_json_phase1`)

The input QA records, augmented with a `question_evaluation` block. Example:

```json
{
  "question_id": "publication_242-publication_242_chapter_6_section_6.1-proc-0-q-0",
  "question":   "What information is considered basic project data?",
  "answer":     "Basic project data refers to ...",
  "citation":   "A single pavement design analysis ...",
  "process_id": "publication_242-publication_242_chapter_6_section_6.1-proc-0",
  "...":        "all other fields from generated-questions_*.json passed through",
  "question_evaluation": {
    "evaluation_answering": {
      "model":      "gpt-oss:120b",
      "evaluation": 0.0
    }
  }
}
```

When `debug = true`, `evaluation_answering` also carries `full_response` — the
raw Ollama response dict, useful for inspecting latency, eval count, and the
model's pre-validation JSON.

### 9.2 CSV (`output_csv` / `output_csv_phase1`)

The driver CSV with the evaluation column appended:

```csv
question_id,evaluation_answering_eval
publication_242-publication_242_chapter_6_section_6.1-proc-0-q-0,0.0
publication_242-publication_242_chapter_6_section_6.1-proc-0-q-1,0.0
...
```

Legacy columns (`neural_network`, `eval_via_citation`, `eval_via_citation_eval`,
`eval`) are dropped if they happen to be present in the input CSV.

Valid values in `evaluation_answering_eval`:

| value  | meaning                                                                  |
| ------ | ------------------------------------------------------------------------ |
| `1.0`  | LLM judged the answer **is** supported by the citation                   |
| `0.5`  | LLM judged it **partial / N/A**                                          |
| `0.0`  | LLM judged the answer is **not** supported                               |
| `-1`   | Skipped — process had no citation (`process_missining_citation` sentinel) |
| empty  | LLM failed every retry, or the question/process id wasn't resolvable      |

### 9.3 Phase 1 vs. final outputs

Phase 1 outputs and "final" outputs carry the same data today because Phase 1
is the only enabled phase. The split exists because the script was originally
designed for a two-phase pipeline (a follow-up "evaluation via citation"
phase, visible as the commented-out branches inside `phase1_process_question`).
That second phase, plus the per-element-type prompts it dispatched to
(`general`, `input`, `output`, `step`), has been removed from the active code
path but the file naming was kept for downstream compatibility.

---

## 10. Architecture and control flow

```
main()
 ├─ read_configuration(args.config)
 ├─ load CSV driver + JSON lookups
 ├─ Client(host=…), AsyncClient(host=…)       # sync client is created but only the async one is used in Phase 1
 │
 ├─ PHASE 1: Evaluation Answering   (if evaluation_answering)
 │   ├─ try to resume from intermediate_json → else output_json_phase1 → else fresh
 │   ├─ build phase1_tasks   = [(qid, qinfo, pinfo), …]    in driver order, skipping done ids
 │   ├─ semaphore = asyncio.Semaphore(max_concurrent_questions)
 │   ├─ asyncio.run(run_phase1):
 │   │     tasks = [phase1_process_question(...) for each]
 │   │     async for done in atqdm(asyncio.as_completed(tasks)):
 │   │         result = await done
 │   │         questions_lookup[qid]["question_evaluation"]["evaluation_answering"] = ...
 │   │         every N items → save_to_json(snapshot, intermediate_json)
 │   ├─ remove intermediate_json   (clean shutdown)
 │   ├─ save_to_json(phase1_results, output_json_phase1)
 │   └─ save phase1 CSV
 │
 ├─ FINAL SAVE
 │   ├─ collect every record with any question_evaluation
 │   ├─ sanitize_question_for_output(...) on each
 │   ├─ save_to_json(final_results, output_json)
 │   └─ save final CSV
 │
 └─ send_pushover_message(...)   # completion notification
```

The single hot path inside `phase1_process_question`:

```
async with semaphore:
    if citation == "No matching citation found":
        return {evaluation: -1}
    prompt = template.replace(…)
    for attempt in 1..retries_number:
        resp = await async_client.chat(model=model, format=schema, temperature=0.0, think=False)
        result = QuestionAnswerEvaluationResponse.model_validate_json(resp.message.content)
        break    # on success
    return {question_id, model, evaluation: result.evaluation}
```

Note that `temperature=0.0` and `think=False` are hard-coded — the run is
intended to be deterministic and to bypass `gpt-oss`-style chain-of-thought
emission so the response body is just the schema-bound JSON.

---

## 11. Notes and gotchas

- **Driver CSV order is authoritative.** Output CSV row order mirrors the
  driver CSV, *not* completion order. The async runner uses
  `asyncio.as_completed` for live progress but the final CSV is reconstructed
  from `questions_lookup` in driver order.
- **`answer` vs `anser` fallback.** The script reads
  `question_info.get("answer", question_info.get("anser", ""))` to tolerate
  an upstream typo in an older generation run. If your data is clean you can
  ignore this.
- **Missing ids are warned, not failed.** Question ids not found in
  `input_questions_data` or whose `process_id` is not in `input_process_data`
  are logged at WARNING and skipped. The output CSV will have empty
  `evaluation_answering_eval` cells for them.
- **`debug = true` blows up the JSON.** `full_response` is the raw Ollama
  response and adds ~kilobytes per question (15 k questions ≈ tens of MB → GB
  range). Use `debug = true` only for spot-checks, not for full runs.
- **Pushover at the end.** The script calls `send_pushover_message` on
  completion. If you don't want a phone notification, comment that call out
  or unset the Pushover env vars used inside `utils.send_pushover_message`.
- **`logs/` must exist.** `loguru` does not auto-create the parent directory
  for `logs/get-categorized-questions-eval-async2.log`. Create it before
  the first run or the logger will crash on import.
- **No CLI override for input/output paths.** All paths come from the TOML.
  Different corpora ⇒ different config files (see the SLURM job, which is
  one-config-per-publication).
- **`Client(host=…)` is unused.** The script constructs both a sync
  `ollama.Client` and an `ollama.AsyncClient`; only the async one is actually
  exercised. The sync client is a vestige from the pre-async version.
- **`format_steps_as_numbered_list` / `generate_process_description`** are
  dead helpers kept from earlier prompt variants that injected structured
  process descriptions. The active prompt only uses the raw `{PROCESS_CITATION}`
  text.
