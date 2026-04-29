# Software Requirements Specification: Chunk Relevance Filter (mode 3 / `random_logical`)

**Feature:** Add an optional relevance-evaluation step between recursive pre-split and logical grouping in mode 3 (`random_logical`) chunking. Each recursive piece is scored `0`/`0.5`/`1` via its own OpenAI chat completion call against the chain-of-thought prompt at `prompts/nemo_eval-02.txt`; the response contains `<scratchpad>...</scratchpad>` reasoning followed by `<json>{score, reason}</json>` which is extracted by regex and post-receipt-validated via Pydantic (`RelevanceJudgment.model_validate_json`). Calls fan out concurrently bounded by `asyncio.Semaphore(self.relevance_concurrency)`. Pieces with `score ≤ 0.5` are excluded from logical grouping. The strict keep policy `score > 0.5` (only `1` survives) is fixed in v1; the `0.5` label is preserved as a distinct value for telemetry and future threshold-tuning. The bulk full-doc-context approach was prototyped and rejected — see §7 R-2 / Decisions.

**Component:** `nvidia-sdq_custom`
**Version:** 0.1 (draft)
**Status:** Proposed
**Companion plan:** `plans/plan-chunk-relevance-filter.md`

---

## 1. Introduction

### 1.1 Purpose
This SRS defines requirements for adding a chunk-level relevance filter to mode 3 chunking. The goal is to remove document boilerplate (sponsor lists, contact blocks, mission statements, disclaimers, references, distribution notices, quality-assurance statements) from logical chunks before they are consumed by downstream artifact extraction and QA generation, so that those stages do not produce noise-derived artifacts and questions. The filter uses a direct OpenAI chat completion call against the chain-of-thought prompt at `prompts/nemo_eval-02.txt`, with the response post-receipt-validated by a Pydantic v2 model (`RelevanceJudgment`) — rather than `BaseLLM.run_chain`. The closed-set score (`0`/`0.5`/`1`) is constrained by a Pydantic `Literal` applied via `model_validate_json()` on the extracted `<json>` block. (`extract_artifacts.py` v4 uses the stricter Structured Outputs `response_format` path; this feature deliberately diverges because the chain-of-thought prompt's tag-wrapping is incompatible with strict-JSON-only output.)

### 1.2 Scope

In scope:
- A new prompt file: `prompts/nemo_eval-02.txt`. Plain text. Contains a `.format()`-style template with exactly one `{CHUNK}` placeholder. Includes: closed-list noise rubric, `0`/`0.5`/`1` score schema, 6 worked examples drawn from the test corpus (covering each score class plus a project-case-study `1` and a boundary-chunk `0.5`), field-by-field semantics for `RelevanceJudgment` (`score`, `reason`), and chain-of-thought output instructions (`<scratchpad>...</scratchpad>` reasoning followed by `<json>...</json>` JSON object).
- New Pydantic v2 model in `_nemo.py`: a module-level `Literal` alias `RELEVANCE_SCORE = Literal[0, 0.5, 1]` (mixed int/float, matching the prompt's exact wording `"Must be exactly 0, 0.5, or 1"`) and a `RelevanceJudgment` BaseModel with fields `score: RELEVANCE_SCORE` and `reason: str`. No wrapper / list model. Used for post-receipt validation via `model_validate_json` on the extracted `<json>` block, NOT as `response_format=...` for OpenAI Structured Outputs (which is incompatible with the chain-of-thought tag-wrapping output format).
- A new `QAGenerator.evaluate_chunks` async method in `_nemo.py` that fans out one plain OpenAI chat completion call per recursive piece via `await asyncio.gather(*[_eval_one(c) for c in chunks])` over `await self.eval_client.chat.completions.create(model="gpt-4o-mini", temperature=0.0, messages=[{"role": "user", "content": prompt_template.format(CHUNK=chunk["text"])}])` calls bounded by `asyncio.Semaphore(self.relevance_concurrency)`. Each response is parsed by extracting the `<json>...</json>` block via regex and validating with `RelevanceJudgment.model_validate_json()`; the `<scratchpad>...</scratchpad>` block is also extracted and stored alongside the score for telemetry. Per-piece exceptions (missing `<json>` block, malformed JSON, off-enum score, network error, refusal) are caught inside `_eval_one` and yield `{"score": 1.0, "reason": "error: <msg>", "scratchpad": None}`. Writes `{chunk_dir}/{doc_id}-relevance.json` once after all per-piece calls complete.
- A new instance attribute `QAGenerator.eval_client: AsyncOpenAI | None` instantiated in `__init__` when `[chunking].relevance_filter == true` and `method == "random_logical"`. `OPENAI_API_KEY` lookup happens at construction; missing key raises `RuntimeError`.
- A new instance attribute `QAGenerator.relevance_concurrency: int`, defaulting to `8`, read from `[chunking].relevance_concurrency`. Bounds the in-flight per-piece eval calls.
- A new free function `group_kept_pieces` in `aisa/parse/chunkers.py` that runs `_llm_split_decisions` over maximal contiguous runs of kept piece indices and returns logical chunks + source-piece provenance referencing original (unfiltered) indices.
- Modifications to `QAGenerator.path2chunks` to (1) become async, (2) drive the recursive pre-split inline (no longer via `HybridLogicalChunker.split`'s `last_recursive_pieces` side-effect), and (3) insert the optional eval + mask-grouping stage inside the existing `if is_hybrid:` branch behind a config flag.
- Updates to existing async callers (`run_chunk_only_pipeline`, `run_sgd_pipeline`, `run_sgd_logical_pipeline`) to `await path2chunks`.
- A new optional config key `[chunking].relevance_filter` (bool, default `false`).
- A new per-doc output file: `{chunk_dir}/{doc_id}-relevance.json` (only written when the filter is on).
- `QAGenerator.__init__` validation that the flag is set with `method = "random_logical"`; otherwise log a warning and skip OpenAI client construction.

Out of scope:
- Mode 1 (`recursive`) and mode 2 (`logical`) — explicitly deferred. `LLMSemanticChunker` is not modified.
- A configurable threshold. The keep policy `score > 0.5` is hard-coded in v1.
- A second-pass / consensus / committee evaluation for borderline pieces.
- Configurable eval model and temperature. v1 hardcodes `gpt-4o-mini` at `temperature=0.0` (matching `extract_artifacts.py` v4).
- Provider portability for the eval call. The eval call is OpenAI-only in v1; routing it through `BaseLLM` (e.g. via `langchain.with_structured_output`) is a follow-up.
- Bulk full-doc eval (all pieces tagged into a single LLM call). Considered and **rejected** during prototyping — the model used document context to over-extend a "References section" classification 4–6 pieces backward into body content. v1 evaluates each recursive piece in its own LLM call. See §7 Decisions and R-2 for the empirical evidence.
- Per-piece result caching (one cache file per recursive piece). v1 keeps the per-doc `{doc_id}-relevance.json` shape; the file is written once after all per-piece calls complete. Per-piece caching could enable mid-doc resume after partial failure but is not needed in v1.
- Schema-version markers on `-relevance.json`.
- A CLI flag (`--relevance-filter`); v1 is TOML-only.
- Migration tooling for existing mode-3 outputs without the filter (they remain valid; the new path is purely additive).
- Manual-review / annotator UI; hand-labeling for verification uses plain JSON files in `data/_test/chunk_test-random-logic/relevance_truth/`.
- Modifications to `extract_artifacts`, `generate_qa_pairs`, `evaluate_qa_pairs`, or any Stage 1 (`run_data_prep_pipeline`) logic.

### 1.3 Definitions
- **Recursive piece** — output of `RecursiveCharacterTextSplitter` at `chunk_size` tokens with `recursive_overlap` overlap, before logical grouping. Stored as the `texts` array in `{doc_id}-chunks.json`.
- **Logical chunk** — output of mode 3 grouping over recursive pieces. Stored as the `texts` array in `{doc_id}-logic-chunks.json`.
- **Relevance score** — value in the closed set `{0, 0.5, 1}` assigned to each recursive piece by the chunk-eval LLM call. `1` = clearly relevant pavement engineering content, `0.5` = unsure / mixed / ambiguous, `0` = clearly noise (matches one of the closed-list categories).
- **Keep policy** — predicate `score > 0.5` applied to relevance scores. Pieces failing this predicate are dropped from logical grouping. v1 only.
- **Kept index** — original recursive piece index `i` such that `score[i] > 0.5`.
- **Kept run** — maximal contiguous sequence of kept indices (e.g. `[0, 1, 2]`, `[4, 5]` in a doc that drops index `3`).
- **Closed-list noise** — categories enumerated in the chunk-eval prompt as automatic `0`: sponsor lists, PI/contact blocks, mission statements, copyright/disclaimer/liability notices, distribution & availability notices, key-words lists, references/bibliography sections, isolated figure/table captions, page headers/footers, ToC fragments, quality assurance statements.
- **Mode 3** — `[chunking].method = "random_logical"`, implemented by `aisa/parse/chunkers.py::HybridLogicalChunker`.
- **Filter on / filter off** — the runtime state determined by `[chunking].relevance_filter` and `method == "random_logical"`. The filter is on iff both conditions hold.
- **Pydantic-validated** — checked at parse time by Pydantic v2 via `RelevanceJudgment.model_validate_json(json_block)` on the extracted `<json>` content. Includes: presence of required fields, scalar types (`str`), Literal enum values for `score`. Validation is post-receipt (after extracting `<json>` from the raw response), NOT at the API boundary — `response_format=<Model>` is not used because the prompt's chain-of-thought tag-wrapping is incompatible with strict JSON-only output. Each per-piece call yields one `RelevanceJudgment` (no list).
- **Soft-validation** — post-receipt handling in Python with logging and substitution (not rejection of the whole response). In v1 this covers all per-piece failure modes: missing `<json>` block in the response, malformed JSON inside the block, off-enum `score` value (e.g. `0.7`) caught by Pydantic, wrong types caught by Pydantic, missing required fields, OpenAI refusal-as-empty-content, network errors, OpenAI API errors. Coverage gaps, duplicate `chunk_id`s, and out-of-range `chunk_id`s are eliminated by the per-piece fan-out construction (every input piece gets its own call; the loop assigns `chunk_id` from input).
- **Validation default** — the substitution applied when one piece's per-piece eval call fails (any soft-validation case above): `{"chunk_id": <id>, "score": 1.0, "reason": "error: <msg>", "scratchpad": None}`. Default `1.0` (keep) preserves recall under the strict `> 0.5` policy.
- **`<json>` block / `<scratchpad>` block** — the two tagged sections of the model's response per the `prompts/nemo_eval-02.txt` template. `<scratchpad>` contains free-form chain-of-thought reasoning; `<json>` contains the structured `{score, reason}` object. Both are extracted by regex (`<json>\s*(.*?)\s*</json>`, `<scratchpad>\s*(.*?)\s*</scratchpad>`) with `re.DOTALL`. The `<scratchpad>` block is preserved in the `-relevance.json` output for telemetry; the `<json>` block is parsed and discarded.
- **Per-piece eval / fan-out** — the v1 evaluation strategy: one OpenAI chat completion call per recursive piece against the `prompts/nemo_eval-02.txt` template (with `{CHUNK}` substituted), dispatched concurrently via `asyncio.gather` and bounded by `asyncio.Semaphore(self.relevance_concurrency)`. Each response is parsed by extracting `<json>` content and post-receipt-validated by Pydantic. Replaces the bulk full-doc-context approach that was prototyped and rejected.
- **In-content citation** — citation appearing inside a content paragraph (e.g. "(Snyder et al. 2018)" mid-sentence). Distinct from a References *section*. In-content citations are content (`1.0`), not noise.

### 1.4 References
- `plans/plan-chunk-relevance-filter.md` — companion implementation plan.
- `docs/logical-chunking.md` — current mode 3 architecture.
- `aisa/parse/chunkers.py` — `HybridLogicalChunker`, `_llm_split_decisions`, `_assemble_with_overlap_trim`, `RecursiveTextChunker`.
- `_nemo.py::QAGenerator` — pipeline orchestration; `path2chunks`, `run_*_pipeline` methods.
- `extract_artifacts.py` and `plans/srs-extract-artifacts-v4-chunk-signals.md` — sibling Pydantic + OpenAI Structured Outputs implementation; this SRS follows the same convention.
- OpenAI Python SDK `AsyncOpenAI().beta.chat.completions.parse(response_format=<BaseModel>)` Pydantic-aware variant of the chat completions API.
- Pydantic v2 `BaseModel` + `Literal[...]` enum support.
- Test corpus: `data/_test/chunk_test-random-logic/doc-chunks_256_random_logical/` (TBF000011, TBF000131).

---

## 2. Overall Description

### 2.1 Product Perspective
Mode 3 currently produces logical chunks that mix domain content with document boilerplate. Empirically (test corpus): TBF000011 piece 0 is wholly head-of-doc boilerplate; TBF000131 pieces 78-83 are tail noise (references, FHWA contact, KEY WORDS, NOTICE, Quality Assurance), and logical chunk 0 mixes the SUMMARY AND DISCLAIMERS section with the introduction. This noise propagates: artifact extraction emits useless metadata extractions, QA generation produces noise-derived questions, and Stage 1 hard-negative mining ranks irrelevant passages.

The relevance filter inserts a fan-out of OpenAI chat completion calls between the recursive pre-split and the logical grouping — one call per recursive piece, dispatched concurrently and bounded by `asyncio.Semaphore(self.relevance_concurrency)` (default `8`). Each call substitutes the chunk text into the `{CHUNK}` placeholder of `prompts/nemo_eval-02.txt` and sends the result as a single user message. The model responds with `<scratchpad>` reasoning followed by `<json>` containing `{score: 0|0.5|1, reason: str}`; the `<json>` block is extracted by regex and validated post-receipt via `RelevanceJudgment.model_validate_json()`. The score is constrained to a closed set by Pydantic on validation (off-enum scores raise `ValidationError`, caught by the per-piece exception handler). Pieces failing the strict keep predicate `score > 0.5` are excluded from logical grouping, so noise cannot bleed into final logical chunks.

A bulk full-doc-context approach was prototyped (all pieces concatenated and tagged in one LLM call) and **rejected**: empirically the model used document structure to over-extend a "References section" classification 4–6 pieces backward into body content (project case studies and conclusions immediately before the `# REFERENCES` heading were misclassified as references). Per-piece eval forces each chunk to be judged on its own content, which is sufficient because boilerplate categories (sponsor lists, references, disclaimers, mission statements, distribution notices, QA statements) are format-recognizable in isolation.

The filter is opt-in via a TOML flag and only active for mode 3. Mode 1 (`recursive`) and mode 2 (`logical`) are unaffected at runtime regardless of the flag value. Existing mode-3 invocations without the flag enabled produce byte-identical output to the pre-feature behavior (the recursive pre-split path is refactored, but the resulting chunk files are unchanged when `kept_indices` is full-coverage). The eval call uses a direct `AsyncOpenAI` client via `chat.completions.create()` — `BaseLLM` is not involved, even when `[llm].provider` is configured for OpenAI; the strict Structured Outputs path (`beta.chat.completions.parse(response_format=...)`) is also not used because the prompt's `<scratchpad>` + `<json>` tag-wrapping is incompatible with strict-JSON-only output. This bypass is deliberate. (`extract_artifacts.py` v4 uses Structured Outputs because its prompt is a pure-JSON-output prompt; the chain-of-thought prompt here makes a different trade.)

### 2.2 User Classes
- **Pipeline operator** — runs `python _nemo.py --chunk-only` or `--sdg` on a corpus. Toggles the filter via `cfg/nemo.toml`. Inspects `-relevance.json` to understand what was dropped and why.
- **Pipeline developer** — extends or tunes the eval prompt, the keep threshold, the chunker integration. Adds worked examples to the prompt. Considers extending the filter to modes 1/2 in a follow-up.
- **Downstream consumer** — reads `-logic-chunks.json` for artifact extraction, QA generation, and hard-negative mining. Sees fewer / cleaner logical chunks when the filter is on; reads `-relevance.json` opportunistically for telemetry.

### 2.3 Operating Environment
Identical to the rest of the pipeline: Python 3.11+, the `aisa.gen.providers` provider registry (OpenAI, Google, Ollama, HuggingFace), prompt library at `./prompts/`, `.env` for `OPENAI_API_KEY` / `GOOGLE_API_KEY`. The eval call adds a direct dependency on the `openai` Python SDK (`AsyncOpenAI` + `beta.chat.completions.parse`) and `pydantic` v2. Both are already available in the project environment (`extract_artifacts.py` v4 uses them; `pydantic==2.11.7` and `openai==1.91.0` per `reqs.txt`).

### 2.4 Constraints
- The filter SHALL only be active when `[chunking].method == "random_logical"`. Other modes SHALL be untouched at runtime regardless of the flag value.
- The filter SHALL be opt-in via `[chunking].relevance_filter` (bool, default `false`).
- The filter SHALL NOT change the schema (key set or value types) of `-chunks.json` or `-logic-chunks.json`. Only the *content* of `source_chunk_ids` lists changes (filtered indices simply absent).
- Score values SHALL be drawn from the closed set `{0, 0.5, 1}` (mixed int/float, matching the prompt's exact wording). Out-of-set values in the model's response SHALL be rejected by `RelevanceJudgment.model_validate_json()` post-receipt (`pydantic.ValidationError`); the per-piece exception handler catches the error and substitutes the validation default.
- Keep threshold SHALL be fixed at `> 0.5` in v1 (only `score == 1.0` survives).
- The eval call SHALL go through `await self.eval_client.chat.completions.create(model="gpt-4o-mini", temperature=0.0, messages=[...])`, dispatched per recursive piece. The `messages` list SHALL contain a single `{"role": "user", "content": <prompt with {CHUNK} substituted>}` entry. No `BaseLLM` involvement for the eval call. The OpenAI Structured Outputs path (`client.beta.chat.completions.parse(response_format=...)`) SHALL NOT be used because the prompt's chain-of-thought tag-wrapping output (`<scratchpad>` + `<json>`) is incompatible with strict JSON-only output.
- The eval model SHALL be `gpt-4o-mini` at `temperature=0.0` in v1 (hardcoded; configurability is OQ-2).
- The eval SHALL fan out one OpenAI call per recursive piece, dispatched concurrently with `asyncio.gather`, bounded by `asyncio.Semaphore(self.relevance_concurrency)`. The per-piece user message SHALL contain only that piece's text — no document context, no neighboring pieces, no tagged-input convention.
- `self.relevance_concurrency` SHALL be read from `[chunking].relevance_concurrency` (int, default `8`) at `__init__`. It SHALL be a positive integer.
- `OPENAI_API_KEY` SHALL be available in the environment when `[chunking].relevance_filter == true` AND `method == "random_logical"`, regardless of `[llm].provider`. `QAGenerator.__init__` SHALL raise `RuntimeError` if the key is missing under those conditions.
- File-cache idempotency SHALL match the rest of the pipeline: existing `-relevance.json` is reused if `self.overwrite == False`. The cache file is written once after all per-piece calls complete.
- A whole-doc exception in `evaluate_chunks` (e.g. `read_prompt` fails) SHALL NOT abort `path2chunks`; `path2chunks` SHALL catch and fall back to keep-all and log one `"CHUNK"` line. Per-piece exceptions are caught inside the per-piece coroutine and yield `score=1.0` for that piece — they do not surface to `path2chunks`.
- Soft-validation defaults SHALL be `score = 1.0` (preserve the piece) — not `0.5` — because `0.5` would silently drop under the strict keep policy. Soft-validation applies only to per-piece exceptions; coverage gaps, duplicates, and out-of-range `chunk_id`s are eliminated by per-piece fan-out construction; out-of-set scores are eliminated by Pydantic enforcement.
- `LLMSemanticChunker` (mode 2 chunker) SHALL NOT be modified by this change.
- Downstream stages (`extract_artifacts`, `generate_qa_pairs`, `evaluate_qa_pairs`, `run_data_prep_pipeline`) SHALL NOT require modification.

### 2.5 Assumptions
- The OpenAI `gpt-4o-mini` model (the hardcoded v1 eval model) has sufficient input budget for one recursive piece (~256 tokens) plus the system prompt (~1K tokens) per call. Easily within any modern model's input budget.
- The OpenAI rate-limit tier for the configured `OPENAI_API_KEY` allows at least `relevance_concurrency` (default 8) parallel `gpt-4o-mini` requests. A typical paid tier supports thousands of RPM and millions of TPM for `gpt-4o-mini`; concurrency 8 is conservative.
- `OPENAI_API_KEY` is set in `.env` or the environment when the filter is on, even if `[llm].provider` is configured for Google or Ollama. The relevance call is OpenAI-only in v1.
- The `openai` Python SDK installed in the environment supports `AsyncOpenAI().beta.chat.completions.parse(response_format=<BaseModel>)`. Per `reqs.txt`, `openai==1.91.0` is pinned and supports this API.
- Pydantic v2 is available (`pydantic==2.11.7` per `reqs.txt`).
- The prompt library directory `./prompts/` exists and is writable.
- `HybridLogicalChunker.recursive` (the inner `RecursiveTextChunker`) is accessible from `path2chunks` for explicit pre-split; alternatively a `RecursiveTextChunker` instance can be constructed in `path2chunks` from the same `chunk_cfg` parameters. Either path produces equivalent recursive pieces.
- Operators force-regenerate stale cache via `self.overwrite = True` when the eval prompt or the Pydantic schema changes. There is no prompt-version tracking in v1.
- Modes 1 and 2 do not need to share the relevance filter in v1; their unit sizes and cache shapes differ.
- The closed-list noise rubric covers the dominant boilerplate categories in the FHWA / Iowa-DOT / CP-Tech-Center tech-brief corpus that motivates the test corpus. Edge cases that emerge in production are addressed by adding worked examples to the prompt; not a code change.

---

## 3. Functional Requirements

### FR-1 New eval prompt file
**FR-1.1** A new prompt SHALL exist at `prompts/nemo_eval-02.txt`. It SHALL contain:
- A task description: score one recursive piece (the chunk substituted into the `{CHUNK}` placeholder) for pavement-engineering relevance on the closed set `{0, 0.5, 1}`.
- A score schema definition: `1` = clearly relevant; `0.5` = unsure / mixed / ambiguous; `0` = clearly noise (matches the closed-list rubric).
- A closed-list noise rubric naming at minimum: sponsor / funder lists; PI / author / contact blocks; mission / vision statements; copyright / disclaimer / liability notices; distribution & availability notices; key-words / index-term lists; references / bibliography sections; isolated figure / table captions; page headers / footers; ToC fragments; quality assurance / endorsement statements.
- A rule distinguishing in-content citations (relevant, `1`) from References *sections* (noise, `0`).
- A "References boundary" note clarifying that content appearing BEFORE a `# REFERENCES` heading (case studies, conclusions, summaries) is content (`1` or `0.5`), not noise.
- 6 worked examples drawn from the test corpus, covering each score class. At minimum: a mixed-head-of-doc `0.5` (sponsors + PI + mission with project title), a pure references list `0`, a pure QA statement `0`, a substantive technical content `1`, a project-case-study `1` (e.g. TBF000131 GDOT I-16 case study) — so the model doesn't confuse case studies near the references section with the references themselves, and a boundary chunk `0.5` (conclusions + figure captions + start of REFERENCES heading).
- Output format instructions: the model SHALL respond with a `<scratchpad>...</scratchpad>` block containing free-form chain-of-thought reasoning, followed by a `<json>...</json>` block containing a JSON object with fields `score` and `reason`.
- Field-by-field semantics for the `RelevanceJudgment` schema (the JSON inside `<json>`):
  - `score` — must be exactly `0`, `0.5`, or `1` (semantics defined above).
  - `reason` — brief explanation, ≤15 words.
- An example output showing both the `<scratchpad>` and `<json>` blocks (with `{{` / `}}` escaping for the JSON braces — see FR-1.2 on `.format()` semantics).
**FR-1.2** The prompt SHALL contain exactly one `{CHUNK}` placeholder. The prompt SHALL be a `.format()`-compatible template; literal `{` and `}` characters in the prompt body (e.g. inside the example output's JSON) SHALL be escaped as `{{` and `}}` per Python `str.format()` conventions. No other placeholders.
**FR-1.3** The prompt SHALL NOT mention the downstream keep threshold. Labels are produced honestly across the three classes; the threshold lives in code.
**FR-1.4** The prompt SHALL NOT use the `<start_chunk_N>...<end_chunk_N>` tagging convention from `_llm_split_decisions` — that convention is for bulk multi-piece input, which this prompt is not. The chunk text is substituted into `{CHUNK}` directly.

### FR-2 New `QAGenerator.evaluate_chunks` method
**FR-2.1** Signature: `async def evaluate_chunks(self, file_path: Path, chunks: dictlist) -> dictlist`.
**FR-2.2** Output file: `{chunk_dir}/{doc_id}-relevance.json` with shape:
```
{
  "doc_id": str,
  "scores": [
    {"chunk_id": int, "score": 0.0 | 0.5 | 1.0, "reason": str, "scratchpad": str | null},
    ...
  ]
}
```
The path SHALL be derived from `self.doc_paths[file_path]` by replacing `-chunks.json` with `-relevance.json`. Score values SHALL be serialized as JSON numbers; the per-piece coroutine casts `judgment.score` (which Pydantic accepts as `int` 0/1 or `float` 0.5) to `float` before assembling the entry, so the file consistently stores `0.0`/`0.5`/`1.0`. The `scratchpad` field is `null` (JSON) when the model didn't include a `<scratchpad>` block or when the per-piece exception fallback fired.
**FR-2.3** Idempotency: if `-relevance.json` exists and `self.overwrite == False`, the method SHALL read the cached file, return its `scores` list verbatim, and emit one `"CHUNK"`-level cache-hit log line. No OpenAI call SHALL be made on a cache hit.
**FR-2.4** The method SHALL fan out one OpenAI chat completion call per recursive piece using `await asyncio.gather(*[_eval_one(c) for c in chunks])`, with `_eval_one` defined as a closure that acquires `asyncio.Semaphore(self.relevance_concurrency)` before calling:
```python
user_content = prompt_template.format(CHUNK=chunk["text"])
completion = await self.eval_client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.0,
    messages=[{"role": "user", "content": user_content}],
)
text = completion.choices[0].message.content or ""
```
where `prompt_template` is the contents of `prompts/nemo_eval-02.txt` loaded once per `evaluate_chunks` invocation via `self.llm.read_prompt("nemo_eval-02")`. `self.eval_client` is the `AsyncOpenAI` instance created by `__init__` per FR-10. The chunk text is substituted into the prompt's `{CHUNK}` placeholder via Python `str.format(CHUNK=chunk["text"])` — `.format()` correctly unescapes `{{` / `}}` in the template body to literal `{` / `}` in the rendered output. The user-message content is the rendered template only — no document context, no neighboring pieces, no system message.
**FR-2.4a** After receiving the response text, `_eval_one` SHALL extract the `<json>...</json>` block via `re.search(r"<json>\s*(.*?)\s*</json>", text, re.DOTALL)`. If no `<json>` block is found, raise `RuntimeError(f"no <json> block in response (head: {text[:200]!r})")`. Otherwise, strip the block content of leading/trailing whitespace and any surrounding code fences (` ```json ` ... ` ``` `).
**FR-2.4b** The extracted JSON text SHALL be validated via `RelevanceJudgment.model_validate_json(json_text)`. On `pydantic.ValidationError` (off-enum score, wrong type, missing field), the exception propagates to FR-2.6's per-piece handler.
**FR-2.4c** The `<scratchpad>...</scratchpad>` block SHALL be extracted via `re.search(r"<scratchpad>\s*(.*?)\s*</scratchpad>", text, re.DOTALL)`. The captured text SHALL be `.strip()`-ped and stored in the entry's `scratchpad` field. If no `<scratchpad>` block is found, the field SHALL be `None`. Missing scratchpad SHALL NOT raise an error; the model is allowed to skip the scratchpad even though the prompt asks for it.
**FR-2.5** The semaphore SHALL be constructed once per `evaluate_chunks` invocation and shared across all per-piece coroutines. The semaphore value SHALL equal `self.relevance_concurrency`.
**FR-2.6** Per-piece exception handling: each `_eval_one` coroutine SHALL catch any exception inside its own `try/except`. Exception sources include: missing `<json>` block (FR-2.4a), malformed JSON inside the `<json>` block (`json.JSONDecodeError` from Pydantic), off-enum `score` (`pydantic.ValidationError`), wrong types or missing fields (Pydantic), network errors, OpenAI API errors, OpenAI refusal-as-empty-content. On any exception the coroutine SHALL return `{"chunk_id": chunk["chunk_id"], "score": 1.0, "reason": f"error: {exc}", "scratchpad": None}` and emit one `"CHUNK"`-level log line naming the doc, chunk_id, and exception. Exceptions SHALL NOT propagate out of `_eval_one` to `asyncio.gather`. Coverage gaps, duplicates, and out-of-range `chunk_id`s are eliminated by construction (one call per input piece; the loop assigns `chunk_id` from input).
**FR-2.7** The returned list SHALL be aligned to the input `chunks` order (the order returned by `asyncio.gather` matches the order of awaitables passed in) and SHALL have length `len(chunks)`. Each entry SHALL be a dict with keys `{"chunk_id": int, "score": float ∈ {0.0, 0.5, 1.0}, "reason": str, "scratchpad": str | None}`.
**FR-2.8** Whole-doc exceptions outside the per-piece coroutines (e.g. `read_prompt` failure, semaphore construction failure) SHALL propagate; `path2chunks` is responsible for catch-and-fallback per FR-4.7.
**FR-2.9** The method SHALL NOT call `self.llm` (the `BaseLLM` instance) for any LLM call. The only `BaseLLM` interaction SHALL be `self.llm.read_prompt("nemo_eval-02")` for prompt loading, which is filesystem I/O and provider-agnostic.
**FR-2.10** The method SHALL write `-relevance.json` exactly once per invocation, after `asyncio.gather` resolves. Partial writes (mid-fan-out) SHALL NOT occur. The file SHALL contain the full `scores` list per FR-2.2.

### FR-9 Pydantic model
**FR-9.1** `_nemo.py` SHALL define a module-level `Literal` alias:
```python
RELEVANCE_SCORE = Literal[0, 0.5, 1]
```
The Literal is mixed-type (int, float, int) by design — it matches the prompt's exact wording (`"Must be exactly 0, 0.5, or 1"`) so that JSON values `0`, `0.0`, `1`, `1.0`, and `0.5` all validate via Pydantic's `==`-based Literal matching.
**FR-9.2** `_nemo.py` SHALL define a Pydantic v2 `BaseModel` `RelevanceJudgment` with fields:
- `score: RELEVANCE_SCORE`
- `reason: str`
Each field MAY carry a `pydantic.Field(description=...)` annotation.
**FR-9.3** `RelevanceJudgment` SHALL be used for **post-receipt validation** via `RelevanceJudgment.model_validate_json(json_text)` on the extracted `<json>` block content (per FR-2.4b). It SHALL NOT be passed as `response_format=...` to `client.beta.chat.completions.parse()` — the prompt's chain-of-thought tag-wrapping output is incompatible with Structured Outputs' strict-JSON-only contract. Validation enforces: required fields populated; `score` value matches the `Literal` set; `reason` is `str`. Validation failures (`pydantic.ValidationError`, `json.JSONDecodeError`) propagate to the per-piece exception handler in FR-2.6.
**FR-9.4** No wrapper / list / `RelevanceResponse` model SHALL be defined. Each per-piece call returns one `RelevanceJudgment` (after JSON extraction and validation); the `chunk_id` association is made by the caller (`_eval_one` reads `chunk["chunk_id"]` from the input). The `scratchpad` field is NOT part of the Pydantic model — it's extracted separately from the response text and attached to the per-piece dict result (per FR-2.4c).
**FR-9.5** The Pydantic model SHALL NOT be exported from a separate module in v1; it lives alongside `QAGenerator` in `_nemo.py`.

### FR-10 AsyncOpenAI client lifecycle and concurrency
**FR-10.1** `QAGenerator.__init__` SHALL conditionally instantiate an `AsyncOpenAI` client when `[chunking].relevance_filter == True` AND `[chunking].method == "random_logical"`. The client SHALL be stored as `self.eval_client: AsyncOpenAI | None`.
**FR-10.2** When the conditions in FR-10.1 are met, `__init__` SHALL look up `OPENAI_API_KEY` via `os.getenv("OPENAI_API_KEY")`. If the value is missing or empty, `__init__` SHALL raise `RuntimeError` with a descriptive message naming `OPENAI_API_KEY` and `[chunking].relevance_filter`.
**FR-10.3** When `[chunking].relevance_filter == True` but `[chunking].method != "random_logical"`, `__init__` SHALL set `self.eval_client = None`, log one `"CHUNK"`-level warning ("relevance_filter ignored: only honored for method='random_logical' (got method=...)"), and SHALL NOT raise even if `OPENAI_API_KEY` is missing.
**FR-10.4** When `[chunking].relevance_filter == False` (default), `__init__` SHALL set `self.eval_client = None` and SHALL NOT consult `OPENAI_API_KEY`.
**FR-10.5** `__init__` SHALL set `self.relevance_concurrency = int(self.chunk_cfg.get("relevance_concurrency", 8))` unconditionally (regardless of filter state). The default is `8`. Implementations MAY validate that the value is a positive integer; non-positive values SHALL fall back to the default with a `"CHUNK"`-level warning.
**FR-10.6** `evaluate_chunks` SHALL assume `self.eval_client is not None` (i.e. the caller `path2chunks` only invokes it when the filter is on under FR-4.3, which implies `eval_client` was constructed). No additional null-check is required.

### FR-3 Mask-aware logical grouping
**FR-3.1** A new module-level function SHALL be added to `aisa/parse/chunkers.py`:
```python
def group_kept_pieces(
    pieces: list[str],
    kept_indices: list[int],
    llm: BaseLLM,
    prompt_template: str,
    window: int,
    stride: int,
    has_overlap: bool,
) -> tuple[list[str], list[list[int]]]
```
**FR-3.2** Algorithm: split `kept_indices` into maximal contiguous runs (consecutive integers). For each run with `len > 1`, build the sub-list `[pieces[i] for i in run]`, call `_llm_split_decisions(llm, prompt_template, sub_pieces, window, stride)`, get sub-list-relative split points, assemble sub-chunks via `_assemble_with_overlap_trim(sub_pieces, sub_splits, has_overlap)`, then map each sub-source-index `i` back to original-index `run[i]`. For each run with `len == 1`, emit the single piece as its own chunk with `source_chunk_ids = [run[0]]` and no LLM call. Concatenate all runs' results.
**FR-3.3** Returned `source_chunk_ids` (the per-chunk lists in the second tuple element) SHALL reference *original* piece indices (the values from `kept_indices`), not sub-list indices.
**FR-3.4** When `kept_indices == list(range(len(pieces)))` and `kept_indices` is non-empty, the function's output SHALL be equivalent to the current `HybridLogicalChunker.split` body operating on those `pieces` with the same `window`/`stride`/`has_overlap`. ("Equivalent" here means the same chunk-text strings and the same source-index lists modulo any internal reordering that `_llm_split_decisions` doesn't have.)
**FR-3.5** When `kept_indices == []`, the function SHALL return `([], [])` and SHALL NOT make any LLM call.
**FR-3.6** `HybridLogicalChunker.split` SHALL remain a public method with unchanged behavior for direct callers. It MAY internally delegate to `group_kept_pieces` with full-coverage `kept_indices`, or retain its current body — implementation choice. The public contract is unchanged.
**FR-3.7** `LLMSemanticChunker` SHALL NOT be modified.

### FR-4 `path2chunks` integration
**FR-4.1** `QAGenerator.path2chunks` SHALL become `async def path2chunks(self, file_path: Path) -> dictlist`. All existing callers SHALL be updated to `await` it: `run_chunk_only_pipeline` (line ~168), `run_sgd_pipeline` (line ~382), `run_sgd_logical_pipeline` (line ~451). Any other callers in the codebase SHALL also be updated.
**FR-4.2** The cache short-circuit at the top of `path2chunks` SHALL be preserved: if `cache_path` (i.e. `-logic-chunks.json` for hybrid, `-chunks.json` otherwise) exists and `self.overwrite == False`, the method SHALL return the cached `texts` without running any new logic — including no relevance call and no logical grouping.
**FR-4.3** Inside the existing `if is_hybrid:` branch (i.e. `method == "random_logical"`):
1. The recursive pre-split SHALL be driven inline (`self.chunker.recursive.split(raw_text)` or an equivalent `RecursiveTextChunker(chunk_size, recursive_overlap).split(raw_text)`), producing `rec_pieces: list[str]`.
2. `rec_chunks` SHALL be built from `rec_pieces` with `chunk_id`/`tokens` per the existing convention and SHALL be written to `{doc_id}-chunks.json` immediately, before any LLM call.
3. If `self.chunk_cfg.get("relevance_filter", False)` is `True`:
   - `await self.evaluate_chunks(file_path, rec_chunks)` SHALL be called.
   - On success: `kept_indices = [c["chunk_id"] for c, s in zip(rec_chunks, scores) if s["score"] > 0.5]`. One `"CHUNK"`-level summary line SHALL be emitted: `<filename>: <kept>/<total> pieces kept (filtered <d>; unsure <h>)` where `d` = count of `score == 0` and `h` = count of `score == 0.5`.
   - On exception: `kept_indices = list(range(len(rec_pieces)))` (fall back to keep-all); one `"CHUNK"`-level log line SHALL describe the failure.
4. Otherwise: `kept_indices = list(range(len(rec_pieces)))`.
5. `group_kept_pieces(rec_pieces, kept_indices, self.llm, prompt_template, hybrid_window, hybrid_stride, has_overlap)` SHALL be called, where `prompt_template = self.chunker.prompt_template`, `hybrid_window/stride` come from `self.chunker.window`/`self.chunker.stride`, and `has_overlap = self.chunker.recursive_overlap > 0`.
6. The returned chunks + `source_chunk_ids` SHALL be written to `{doc_id}-logic-chunks.json` per the existing schema (with `chunk_id` per the new chunk's position in the list, `tokens` recomputed via `get_token_count`, and `source_chunk_ids` taken verbatim from `group_kept_pieces`).
**FR-4.4** Non-hybrid branches (`method != "random_logical"`) SHALL be untouched. The recursive-mode path and the logical-mode path retain their current behavior.
**FR-4.5** `-chunks.json` SHALL continue to record all recursive pieces unchanged regardless of filter state. The filter is a property of the `-logic-chunks.json` derivation, not the `-chunks.json` storage.
**FR-4.6** `relevance_filter = true` set with `method != "random_logical"` SHALL be ignored at runtime per FR-10.3.
**FR-4.7** Eval-call exception handling: if `await self.evaluate_chunks(...)` raises, `path2chunks` SHALL log one `"CHUNK"`-level line, set `kept_indices = list(range(len(rec_pieces)))`, and continue. The document is processed as if the filter were off; `-relevance.json` is not written.

### FR-5 Configuration
**FR-5.1** A new optional key SHALL be honored: `[chunking].relevance_filter` (bool, default `false`).
**FR-5.2** A missing or `false` value SHALL preserve current behavior end-to-end (no eval call, no `-relevance.json`, byte-identical chunk files modulo the inline-driven recursive split which produces identical pieces).
**FR-5.3** A `true` value with `[chunking].method != "random_logical"` SHALL be ignored at runtime per FR-4.6.
**FR-5.4** The committed `cfg/nemo.toml` SHALL NOT enable the flag by default. Operators set it explicitly.
**FR-5.5** No CLI flag SHALL be added in v1.

### FR-6 Output artifact
**FR-6.1** When the filter is on, a new file SHALL be written: `{chunk_dir}/{doc_id}-relevance.json` per FR-2.2.
**FR-6.2** The `scores` list length SHALL equal `len(rec_chunks)`.
**FR-6.3** `chunk_id` values in `-relevance.json` SHALL match the `chunk_id` values in the corresponding `-chunks.json` (same recursive piece indexing).
**FR-6.4** When the filter is off, `-relevance.json` SHALL NOT be written.
**FR-6.5** `-relevance.json` SHALL be readable as JSON via `aisa.utils.files.read_json`.

### FR-7 Behavior preservation
**FR-7.1** The schema (top-level keys, value types) of `-chunks.json` SHALL NOT change.
**FR-7.2** The schema of `-logic-chunks.json` SHALL NOT change. Each entry continues to have `text`, `chunk_id`, `tokens`, `source_chunk_ids`. `source_chunk_ids` continues to reference original recursive piece indices; filtered-out pieces are simply absent from any logical chunk's `source_chunk_ids`.
**FR-7.3** No downstream stage (`extract_artifacts`, `generate_qa_pairs`, `evaluate_qa_pairs`, `run_data_prep_pipeline`) SHALL require modification to consume the filtered `-logic-chunks.json`.
**FR-7.4** Modes 1 and 2 outputs SHALL be byte-identical with and without the flag set, since the flag is ignored.
**FR-7.5** Mode 3 output with `relevance_filter = false` SHALL be byte-identical to the pre-feature output for the same recursive pieces. (The recursive pre-split is driven inline rather than via `HybridLogicalChunker.split`'s side-effect; the resulting pieces are equivalent because both paths use the same `RecursiveCharacterTextSplitter` configuration.)

### FR-8 Logging
**FR-8.1** When the filter is on, `path2chunks` SHALL emit one `"CHUNK"`-level summary line per document, formatted: `<filename>: <kept>/<total> pieces kept (filtered <d>; unsure <h>)`.
**FR-8.2** Validation defaults (FR-2.6) SHALL emit one `"CHUNK"`-level warning per substituted entry, naming the doc and chunk_id.
**FR-8.3** Cache hits on `-relevance.json` SHALL emit one `"CHUNK"`-level cache-hit log line.
**FR-8.4** Mode-mismatch warnings (FR-4.6) SHALL be emitted at `QAGenerator.__init__` time, exactly once per instance.
**FR-8.5** Eval-call exceptions caught by `path2chunks` (FR-4.7) SHALL emit one `"CHUNK"`-level line per occurrence describing the failure.

---

## 4. Non-Functional Requirements

### NFR-1 Backward compatibility
- With `[chunking].relevance_filter = false` (default), the pipeline SHALL produce byte-identical output for `-chunks.json` and `-logic-chunks.json` to the pre-feature behavior on mode 3 inputs. Both files shall match the existing fixtures in `data/_test/chunk_test-random-logic/doc-chunks_256_random_logical/` modulo any LLM nondeterminism in the existing logical-grouping call.
- Modes 1 (`recursive`) and 2 (`logical`) SHALL produce byte-identical output regardless of the flag value.
- Existing `-chunks.json` and `-logic-chunks.json` files in production output directories SHALL remain valid input to downstream stages without migration.
- No deletion of v1/v2/v3 prompts or other historical files.

### NFR-2 Determinism
With `temperature=0.0` on the configured eval client, repeated runs over identical input SHOULD produce identical scores, modulo OpenAI's internal nondeterminism (small variations persist even at temperature 0). Validation defaults (FR-2.6), `kept_indices` derivation, contiguous-run construction, and cache file naming are deterministic. The eval call's prompt rendering (`prompt_template.format(CHUNK=chunk["text"])`), regex extraction of `<json>` and `<scratchpad>` blocks, and message construction are deterministic.

### NFR-3 Observability
The `-relevance.json` file SHALL preserve every score and reason, providing full traceability for "why was this chunk dropped?" queries. Cache hits, validation defaults, mode mismatches, and eval-call exceptions are each logged once per occurrence at `"CHUNK"` level.

### NFR-4 Performance envelope
- Cold run: N additional LLM calls per document (one per recursive piece), bounded by `Semaphore(self.relevance_concurrency)`. For TBF000131 (84 pieces) at gpt-4o-mini, system prompt (~1K tokens) + chunk text (~256 tokens) per call → ~109K input tokens total per doc → ~$0.015/doc. ~5× the bulk-eval cost; absolute cost remains low (~$1.50 / 100 docs).
- Wall-time impact: at concurrency 8, 84 pieces fan out across ~11 batches → ~5–10 s per doc. The eval call is a small fraction of total chunk-stage time on cold runs.
- Warm run: cache hits on `-relevance.json` eliminate all eval calls entirely; latency overhead is one extra `Path.exists()` + `read_json` per doc.
- Per-piece input budget: each call sees one chunk (~256 tokens) plus the system prompt (~1K tokens). Easily within `gpt-4o-mini`'s 128K context budget — the bulk-approach budget concern (long docs not fitting in one call) is eliminated by per-piece fan-out.

### NFR-5 Error containment
- `evaluate_chunks` exceptions SHALL NOT abort `path2chunks`; the method falls back to keep-all per FR-4.7.
- Malformed model output (per-entry) SHALL NOT abort the document; per-entry validation defaults (FR-2.6) preserve the run.
- A wholesale invalid response (not a list, not parseable as JSON) SHALL trigger validation defaults for *all* `chunk_id`s with `score = 1`, equivalent to keep-all for that document, plus one `"CHUNK"` log line per substitution.
- A failed eval call on one document SHALL NOT affect processing of other documents in the same pipeline run.

### NFR-6 Schema stability
The chunking output schemas (`-chunks.json`, `-logic-chunks.json`) SHALL remain strictly invariant: same top-level keys, same per-entry keys, same value types. The new `-relevance.json` is purely additive and not read by any existing stage.

### NFR-7 Test-corpus compatibility
The feature SHALL be testable end-to-end on `data/_test/chunk_test-random-logic/` with TBF000011 and TBF000131 as primary fixtures. Hand-labeled truth files at `data/_test/chunk_test-random-logic/relevance_truth/{doc_id}-relevance-truth.json` SHALL be the precision/recall reference.

---

## 5. Interfaces

### 5.1 CLI interface
Unchanged. The existing flags (`--chunk-only`, `--sdg`, `--sdg-logical`, `--prep`, `--cfg`, `--input_dir`, `--output_dir`) work identically. The filter is enabled via TOML, not CLI.

### 5.2 Python interface
```python
# In _nemo.py — module-level Pydantic model
from typing import Literal
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

RELEVANCE_SCORE = Literal[0, 0.5, 1]

class RelevanceJudgment(BaseModel):
    score: RELEVANCE_SCORE = Field(description="1=relevant; 0.5=unsure/mixed; 0=closed-list noise")
    reason: str = Field(description="Brief explanation, ≤15 words")

# In _nemo.py / QAGenerator
class QAGenerator:
    eval_client: AsyncOpenAI | None             # set in __init__ per FR-10
    relevance_concurrency: int                  # set in __init__ per FR-10.5

    async def evaluate_chunks(self, file_path: Path, chunks: dictlist) -> dictlist: ...
    # Fans out per-piece via asyncio.gather, bounded by Semaphore(self.relevance_concurrency).
    # Each call uses chat.completions.create() against prompts/nemo_eval-02.txt with {CHUNK} substituted.
    # <json> block extracted by regex and validated post-receipt via RelevanceJudgment.model_validate_json().
    # <scratchpad> block extracted and attached to the per-piece dict.
    # Returns: list aligned to chunks:
    #   [{"chunk_id": int, "score": 0.0|0.5|1.0, "reason": str, "scratchpad": str | None}, ...]

    async def path2chunks(self, file_path: Path) -> dictlist: ...   # was sync; now async

# In aisa/parse/chunkers.py
def group_kept_pieces(
    pieces: list[str],
    kept_indices: list[int],
    llm: BaseLLM,
    prompt_template: str,
    window: int,
    stride: int,
    has_overlap: bool,
) -> tuple[list[str], list[list[int]]]: ...
```

### 5.3 File interface
Inputs:
- `{input_dir}/*.md` — unchanged; markdown source.
Outputs (mode 3, filter on):
- `{chunk_dir}/{doc_id}-chunks.json` — recursive pieces; schema unchanged (FR-7.1).
- `{chunk_dir}/{doc_id}-relevance.json` — NEW; schema per FR-2.2 / FR-6.
- `{chunk_dir}/{doc_id}-logic-chunks.json` — logical chunks; schema unchanged (FR-7.2). When filter is on, `source_chunk_ids` only contains kept indices.

Outputs (mode 3, filter off):
- `{chunk_dir}/{doc_id}-chunks.json`, `{chunk_dir}/{doc_id}-logic-chunks.json` — unchanged from pre-feature behavior.
- No `-relevance.json` written.

Outputs (modes 1 and 2):
- Unchanged from pre-feature behavior. No `-relevance.json` written even if `relevance_filter = true` is set.

### 5.4 Configuration interface
```toml
[chunking]
method = "random_logical"
chunk_size = 256
recursive_overlap = 50
hybrid_window = 8
hybrid_stride = 6
relevance_filter = true       # NEW; default false; only honored when method == "random_logical"
relevance_concurrency = 8     # NEW; default 8; bounds in-flight per-piece eval calls (any positive int)
```

### 5.5 Environment interface
- `OPENAI_API_KEY` SHALL be set in `.env` or the environment when `[chunking].relevance_filter == true` AND `[chunking].method == "random_logical"`. The resolution is via `os.getenv("OPENAI_API_KEY")` directly in `QAGenerator.__init__`, independent of `aisa/gen/providers.py`'s resolution for `BaseLLM`. The eval call uses a direct `AsyncOpenAI` client.
- All other env vars (`OPENAI_API_KEY` / `GOOGLE_API_KEY` resolution for `BaseLLM`) SHALL behave identically to today.

### 5.6 Prompt interface
- File: `prompts/nemo_eval-02.txt`. Plain text. `.format()`-compatible template with exactly one `{CHUNK}` placeholder. Literal `{` / `}` characters in the prompt body (e.g. inside the example output's JSON) are escaped as `{{` / `}}`. The chunk text is substituted into `{CHUNK}` and the rendered string is sent as a single *user* message.

---

## 6. Acceptance Criteria

- **AC-1** `prompts/nemo_eval-02.txt` exists. The file contains: a task description; the closed-list noise rubric (with at minimum the 11 categories named in FR-1.1); the `0`/`0.5`/`1` score schema with definitions; the in-content-citation vs. References-section distinction; the References-boundary clarification; 6 worked examples drawn from the test corpus, covering each score class plus a project-case-study `1` and a boundary-chunk `0.5`; field-by-field semantics for `RelevanceJudgment` (`score`, `reason`); chain-of-thought output instructions (`<scratchpad>` then `<json>`). The file SHALL contain exactly one `{CHUNK}` placeholder (FR-1.2). The file SHALL NOT contain `<start_chunk_N>` tagging conventions (FR-1.4) or any other placeholder. Verified by grep.
- **AC-2** `QAGenerator.evaluate_chunks` exists with signature `async def evaluate_chunks(self, file_path: Path, chunks: dictlist) -> dictlist`. Implementation fans out per-piece via `asyncio.gather` over `await self.eval_client.chat.completions.create(model="gpt-4o-mini", temperature=0.0, messages=[{"role": "user", "content": prompt_template.format(CHUNK=chunk["text"])}])` calls bounded by `asyncio.Semaphore(self.relevance_concurrency)`. Each response is parsed by extracting `<json>...</json>` content via `re.search` and validated via `RelevanceJudgment.model_validate_json`; `<scratchpad>...</scratchpad>` content is also extracted. Cold run on a test fixture returns a list of length `len(chunks)`; each entry has `int` `chunk_id`, `float` `score ∈ {0.0, 0.5, 1.0}`, `str` `reason`, `str | None` `scratchpad`.
- **AC-2a** Pydantic model imports cleanly: `from _nemo import RelevanceJudgment, RELEVANCE_SCORE`. `RelevanceJudgment.model_json_schema()` includes `properties.score.enum` listing `[0, 0.5, 1]` (or equivalent — Pydantic may render mixed-type Literal values in either order). `RELEVANCE_SCORE` is a `typing.Literal` of those three values. There is no `RelevanceResponse` / `RelevanceItem` / list-wrapper class in `_nemo` (verified by `hasattr` check). The model is NOT passed as `response_format=...` anywhere in the codebase (verified by grep for `response_format=RelevanceJudgment`).
- **AC-2b** `QAGenerator.__init__` with `[chunking].relevance_filter = true` and `[chunking].method = "random_logical"` sets `self.eval_client` to an `AsyncOpenAI` instance (verified by `isinstance(qagen.eval_client, AsyncOpenAI)`). With either condition false, `self.eval_client is None`.
- **AC-2c** `QAGenerator.__init__` with the filter on and `OPENAI_API_KEY` unset raises `RuntimeError` whose message contains `"OPENAI_API_KEY"`. With the filter off, missing key is tolerated and no exception is raised.
- **AC-2d** `self.relevance_concurrency` defaults to `8` when `[chunking].relevance_concurrency` is not set, and reads the configured int when set. The semaphore in `evaluate_chunks` honors this value (verified by counting concurrent in-flight `parse` calls via a stub).
- **AC-3** Cold run on TBF000011 (`relevance_filter = true`) writes `data/_test/chunk_test-random-logic/.../TBF000011_UKN000-relevance.json` with 6 entries; runs the eval call exactly once (verified via `BaseLLM.info` token counts or LLM-call log telemetry); produces `-logic-chunks.json` from kept pieces.
- **AC-4** Cold run on TBF000131 (`relevance_filter = true`) writes `-relevance.json` with 84 entries. Pieces 78, 79, 80, 81, 82, 83 receive `score = 0`. The resulting `-logic-chunks.json` has no logical chunk whose `source_chunk_ids` contains any of `{78, 79, 80, 81, 82, 83}`. Verified by parsing `-logic-chunks.json` and checking the union of all `source_chunk_ids` lists is disjoint from the noise set.
- **AC-5** Idempotency: a second invocation with `self.overwrite == False` makes zero LLM calls and produces byte-identical output across `-chunks.json`, `-relevance.json`, `-logic-chunks.json`. Verified by comparing file mtimes (unchanged) and content hashes.
- **AC-6** Filter off (default) on the test corpus produces `-chunks.json` and `-logic-chunks.json` that match the existing fixtures byte-for-byte modulo LLM nondeterminism in the logical-grouping call. No `-relevance.json` is written. Verified by file-existence check and field-set comparison via Python.
- **AC-7** Mode 1 (`recursive`) and mode 2 (`logical`) cold runs with `relevance_filter = true` set: outputs identical to outputs with `relevance_filter = false`; no `-relevance.json` written; exactly one `"CHUNK"`-level warning ("relevance_filter ignored: ...") emitted at `QAGenerator.__init__` time per instance.
- **AC-8** `group_kept_pieces` smoke test: with `kept_indices == list(range(N))` produces output equivalent to the current `HybridLogicalChunker.split` body. With a synthetic gap (e.g. `kept_indices = [0, 1, 2, 4, 5, 6, 7]` for N=8), no returned `source_chunk_ids` crosses the gap; output contains at least one chunk whose sources ⊆ `{0, 1, 2}` and another whose sources ⊆ `{4, 5, 6, 7}`. Verified by direct function call in a test harness.
- **AC-9** `group_kept_pieces` empty input: `kept_indices == []` returns `([], [])` and makes zero LLM calls.
- **AC-10** Per-piece exception fallback covers all post-receipt failure modes. Verified by stubbing the eval client to:
  - Return text with no `<json>` block → fallback `{"score": 1.0, "reason": "error: no <json> block ...", "scratchpad": None}`.
  - Return text with malformed JSON inside `<json>` → fallback (`json.JSONDecodeError`).
  - Return text with off-enum score (`"score": 0.7`) → fallback (`pydantic.ValidationError`).
  - Return text with missing `reason` field → fallback (`pydantic.ValidationError`).
  - Raise `RuntimeError` directly → fallback.
  In all cases: one `"CHUNK"`-level log line is emitted naming the doc, chunk_id, and exception; the failed piece's `score == 1.0` so it remains in `kept_indices`; other pieces complete normally. Missing-`chunk_id` coverage gaps, duplicate `chunk_id`s, and out-of-range `chunk_id`s are infeasible by construction (per-piece fan-out) and SHALL NOT need application-level handling.
- **AC-10a** Scratchpad capture: when the model includes a `<scratchpad>...</scratchpad>` block, the extracted text (post-`.strip()`) is stored in the entry's `scratchpad` field. When the model omits the scratchpad, `scratchpad is None`. When the per-piece exception fallback fires, `scratchpad is None`. Verified by inspection of cold-run `-relevance.json` for both fixtures.
- **AC-11** Eval-call exception handling: forcing an exception in `evaluate_chunks` (e.g. via monkeypatch raising `RuntimeError`) does not abort `path2chunks`; the document is processed with all pieces kept; one `"CHUNK"`-level log line describing the failure is emitted; no `-relevance.json` is written for that document.
- **AC-12** Async correctness: running `python _nemo.py --chunk-only --cfg cfg/nemo.toml` (with `relevance_filter = true`) produces no `RuntimeWarning: coroutine ... was never awaited` warnings. All `path2chunks` callers (`run_chunk_only_pipeline`, `run_sgd_pipeline`, `run_sgd_logical_pipeline`) await the coroutine.
- **AC-13** Schema stability: the JSON top-level key set and per-entry key set of `-chunks.json` and `-logic-chunks.json` is identical regardless of filter state. Only `source_chunk_ids` *content* differs. Verified by `set(json.load(...).keys())` comparison.
- **AC-14** Logging counts: a filter-on cold run on a 2-doc corpus with no validation defaults and no exceptions emits exactly 2 summary lines (one per doc), 0 cache-hit lines, 0 validation warnings, 0 exception lines. A second invocation emits exactly 2 cache-hit lines per doc (one for `-relevance.json`, one for the existing `-logic-chunks.json` short-circuit).
- **AC-15** `cfg/nemo.toml` parses correctly with the new key present and absent. Both forms produce a valid `chunk_cfg` dict that `QAGenerator.__init__` consumes without error.

Spot-checks against truth (manual review):

- **SC-1** TBF000011 with filter on: chunk 0 receives `score ∈ {0, 0.5}`; chunks 1-5 receive `score = 1`. The resulting `-logic-chunks.json` has logical chunks whose `source_chunk_ids` ⊆ `{1, 2, 3, 4, 5}`. The text of any returned logical chunk does not contain "Iowa Highway Research Board", "Principal Investigator", "vern@iastate.edu", or "The mission of the National Concrete Pavement Technology Center".
- **SC-2** TBF000131 with filter on: a manual scan of the resulting `-logic-chunks.json` shows zero references-list, FHWA-disclaimer, contact-info, availability-notice, or quality-assurance content in any `texts[].text`. Specifically: no entry contains "FHWA-HIF-22-020", "AGREEMENT OFFICER'S REPRESENTATIVE", "QUALITY ASSURANCE STATEMENT", or strings of references in the form `Author. <Year>. *Title*. Journal, Vol. X, pp. Y-Z`.
- **SC-3** False-negative budget on `score=1` (the failure mode under strict `> 0.5`): zero `truth=1 → model=0` errors on the test corpus; minimize `truth=1 → model=0.5` errors via prompt iteration. Each mismatch tracked in a notes file.
- **SC-4** At least one piece in TBF000131 that contains an in-content citation (e.g. "(Snyder et al. 2018)") receives `score = 1`, demonstrating the in-content-vs-section distinction works.

---

## 7. Risks and Open Questions

### 7.1 Risks

- **R-1 False negatives on borderline content.** The strict `> 0.5` policy drops `0.5`-labeled pieces. Mixed pieces (boilerplate + intro in the same chunk, e.g. TBF000131 piece 0) are likely cut entirely. Mitigation: prompt iteration on the test corpus before running on full corpus; the `-relevance.json` file preserves every reason so dropped pieces can be reviewed; spot-check SC-1/SC-2 require manual review.
- **R-2 Bulk-context drift (mitigated by design choice).** The bulk full-doc-context approach prototyped earlier mislabeled 4-6 pieces of body content as "References section" because the model extended structural inferences across the full document. Per-piece eval (v1) prevents this exact failure mode by construction — each piece is judged on its own content, with no neighboring pieces in the prompt. Residual risk: the per-piece judgment may be wrong on individual ambiguous pieces (e.g. a chunk with only a section heading and minimal content). Mitigation: 4–6 worked examples in the prompt span all score classes and an explicit project-case-study `1` example; SC-3 enforces zero `truth=1 → model=0` errors on the test corpus.
- **R-3 Model overconfidence on `0`.** The model may aggressively label content as noise. Pydantic post-receipt validation constrains *score values* (rejects off-enum) but does not constrain the model's *judgment* about what counts as relevant. Mitigation: closed-list rubric in the prompt restricts what counts as `0`; chain-of-thought `<scratchpad>` reasoning in the prompt encourages the model to articulate its judgment before scoring (and is captured for inspection); per-piece exception default is `1.0` (preserves piece on call failure); SC-3 enforces zero `truth=1 → model=0` errors on the test corpus.
- **R-4 Cache invalidation friction.** A change to the eval prompt or the Pydantic schema requires regenerating all `-relevance.json` files. There is no prompt-version or schema-version tracking. Mitigation: documented; operators delete the relevance files and rerun with `self.overwrite = True`. Same friction as the rest of the pipeline (existing chunking output, artifact files, etc.).
- **R-5 OpenAI cost on large corpora.** N eval calls per doc (one per recursive piece) multiplies by corpus size. At gpt-4o-mini, ~$0.015/doc for an 84-piece doc (~5× the bulk approach due to system prompt amortization loss). For 100 docs: ~$1.50. Still cheap absolute. The hardcoded `gpt-4o-mini` keeps cost independent of `[llm].model`. Mitigation: cache hits dominate after first run; OQ-2 exposes a configurable model if needed.
- **R-5d Rate-limit pressure under fan-out.** Per-piece fan-out can hit OpenAI rate limits on large corpora (especially if multiple docs are processed sequentially in a tight loop). At concurrency 8, peak in-flight requests per doc are 8; sustained throughput is bounded by the slowest call in the batch. Mitigation: `Semaphore(self.relevance_concurrency)` caps parallelism; default `8` is well below `gpt-4o-mini`'s typical-tier RPM limits; `[chunking].relevance_concurrency` exposes the knob if a different tier needs tuning.
- **R-5a OpenAI lock-in for the eval call.** The eval call is OpenAI-only by construction; there's no fallback to Google / Ollama / etc. for relevance scoring even if those providers are configured for `BaseLLM`. Mitigation: scope is documented (FR-9 / FR-10 / §2.4); OQ-3 covers provider portability via `langchain.with_structured_output` (which all major providers now support).
- **R-5b Malformed-output handling.** Without OpenAI Structured Outputs at the API boundary, the model could in principle emit (a) no `<json>` block, (b) malformed JSON inside `<json>`, (c) off-enum `score`, (d) wrong field types, or (e) missing required fields. All such cases are caught by the per-piece exception handler in FR-2.6 and yield the validation-default `score=1.0`. Mitigation: the prompt's worked examples are explicit about the output format; `gpt-4o-mini` at `temperature=0.0` is highly compliant; the per-piece fallback ensures one bad response can't poison the doc.
- **R-5c Refusal handling per-piece.** OpenAI may return a refusal (e.g. content policy) on a particular call; `completion.choices[0].message.content` is empty (or contains a refusal explanation without `<json>` tags). The per-piece coroutine treats this as an exception per FR-2.6 (no `<json>` block found): yields `score=1.0` for that piece, logs once, continues. Other pieces are unaffected. The corpus is technical pavement-engineering content; refusals are not expected but the per-piece fallback covers them safely without aborting the document.
- **R-6 Eval wall-time latency.** Adds N LLM round-trips per doc on cold runs, bounded by `Semaphore(8)`. For TBF000131 (84 pieces) at concurrency 8: ~11 batches → ~5–10 s. For SDG pipelines that already make many calls per doc, the marginal latency is small. Cache hits eliminate it on warm runs.
- **R-7 Refactor risk in `path2chunks`.** The recursive pre-split is moved out of `HybridLogicalChunker.split`'s side-effect path and driven inline. Behavior with `kept_indices` set to full-coverage is intended to be equivalent, but a subtle difference (e.g. how `last_recursive_pieces` was used previously) could leak. Mitigation: AC-6 byte-comparison with existing fixtures catches drift; SC tests cover end-to-end content.
- **R-8 Mode-2 / mode-1 users may want the same filter.** v1 is mode-3 only. Adopting it for other modes requires per-mode design (different unit sizes, different cache shapes); deferred. Risk: feature creep pressure. Mitigation: the deferral is documented explicitly; `evaluate_chunks` is reusable infrastructure for those modes when adopted.
- **R-9 Async migration of `path2chunks`.** Making the method async is a small but pervasive change. A missed `await` in some caller would cause silent coroutine-never-awaited warnings. Mitigation: AC-12 explicitly checks for `RuntimeWarning`; existing callers are enumerated in FR-4.1.

### 7.2 Open questions (non-blocking)

- **OQ-1 Configurable threshold.** Expose `[chunking].relevance_threshold` (float, default `0.5`, applied as `score > threshold`) so operators can choose `>= 0.5` (lax) vs `> 0.5` (strict) without re-running the eval. Defer until the strict default is shown to be too aggressive on real data.
- **OQ-2 Configurable eval model and temperature.** Expose `[chunking].relevance_model` (str, default `"gpt-4o-mini"`) and `[chunking].relevance_temperature` (float, default `0.0`). Defer until a use case emerges — most operators won't need to override `gpt-4o-mini`.
- **OQ-3 Provider portability for the eval call.** Route through `BaseLLM` via `langchain.with_structured_output(RelevanceJudgment)` (supported by `ChatOpenAI`, `ChatGoogleGenerativeAI`, `ChatAnthropic`) so that `[llm].provider = "google"` / `"anthropic"` users don't need a separate `OPENAI_API_KEY`. Defer until requested; v1 mirrors `extract_artifacts.py` v4's direct-OpenAI choice.
- **OQ-3a Bulk eval revisit with stronger model.** Per-piece eval was chosen because `gpt-4o-mini` over-extended the references classification under bulk full-doc-context. A more capable model (`gpt-4o`, `gpt-4.1`) might handle bulk correctly and recover cross-chunk context (1 round-trip per doc instead of N). Defer until a model upgrade is on the table; v1 commits to per-piece for the prototyped model.
- **OQ-3b Migrate other `_nemo.py` LLM calls to Pydantic + Structured Outputs.** Currently `extract_artifacts`, `generate_qa_pairs`, and `evaluate_qa_pairs` use `BaseLLM.run_chain` + manual JSON; only `evaluate_chunks` (this feature) uses Pydantic. Unifying on one pattern would simplify the file. Defer because those three methods produce open-shaped content (variable-length artifact lists, free-text fields, prose answers) where a fixed Pydantic schema would constrain rather than help. Revisit if drift in their JSON output causes recurring breakage.
- **OQ-3c Per-piece result caching for mid-doc resume.** v1 writes `-relevance.json` once after all per-piece calls complete; if `path2chunks` is interrupted mid-fan-out, the next run re-evaluates every piece. Per-piece caching (one cache file per piece, or atomic incremental updates to the per-doc file) would enable resume without re-spending. Defer until a long-running corpus run shows the resume case matters.
- **OQ-4 Schema-version marker on `-relevance.json`.** Add a `prompt_version` or `schema_version` field for downstream traceability. Defer until cache invalidation becomes a real issue.
- **OQ-5 Inline citation handling.** The prompt distinguishes "References *section*" from "in-content citation" via prose. If the model misclassifies in-content citations as `0`, add a worked example to the prompt. Defer until measured (SC-4 enforces correctness on the test corpus).
- **OQ-6 Mode 1 / mode 2 adoption.** Both modes can adopt the same filter with different unit sizes (mode 1: final pieces; mode 2: 50-tok pre-pieces). Defer to a future task with its own SRS.
- **OQ-7 Filter behavior in `--chunk-only` runs.** Currently `path2chunks` is the entry point for all pipelines including `run_chunk_only_pipeline`, so the filter runs there as well (and writes `-relevance.json`). Operators may want a way to run chunking without the filter (e.g. for cache pre-warming). Workaround: toggle the TOML flag for the `--chunk-only` invocation. No code change unless requested.
- **OQ-8 Hand-labeled truth maintenance.** The `relevance_truth/{doc_id}-relevance-truth.json` files require manual upkeep when the corpus changes. Defer; truth files are scoped to fixtures, not production.
- **OQ-9 Multi-LLM consensus / committee for borderline pieces.** Run two models and require agreement to drop. Higher confidence; higher cost. Defer until single-model false-positive rate is measured.
- **OQ-10 Surface relevance scores to downstream stages.** `extract_artifacts`, `generate_qa_pairs` could prefer high-confidence pieces (`score == 1`) over `0.5` if the threshold is later relaxed. v1 doesn't expose this; downstream stages only see kept pieces post-grouping. Defer.
