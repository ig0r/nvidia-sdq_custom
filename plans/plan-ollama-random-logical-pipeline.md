# Plan: Ollama support for the whole `random_logical` pipeline (local debug, `gpt-oss:20b`)

**Companion SRS:** `plans/srs-ollama-random-logical-pipeline.md`
**Status:** Proposed
**Target:** local Ollama `gpt-oss:20b` @ `http://localhost:11434`; corpus `./rawdata-pubs/parsed-specs`; output `./data/specs_20260516`

## Why

Run the full `random_logical` SDG pipeline end-to-end on local Ollama (no OpenAI/Gemini) to produce
a set of LLM-evaluated questions from the PennDOT spec corpus. Audit shows stages 3–4 are already
Ollama-native (config only); two code paths are hard-bound to OpenAI and must be refactored.

## Pipeline

```
1. _nemo.py --sdg-logical --cfg cfg/nemo_specs.toml  → *-chunks / *-logic-chunks / *-logic-ctx (+ *-relevance)
2. extract_artifacts.py --cfg extract_artifacts_specs.toml → *-logic-artifacts.json
3. generate-qa.py --config generate-qa_specs.toml    → generated-questions.json (+ _wo_context, .csv)
4. self-check/self-check-qa.py --config ./self-check-qa_specs.toml → self-check-qa-results.json  ← deliverable
```

## Decisions (user)

- User **renames** `rawdata-pub` → `rawdata-pubs`; `data_dir` in `nemo_specs.toml` stays as-is.
- Relevance filter is **refactored to Ollama** and kept `relevance_filter = true`.
- Separate `_specs` config files (preserve existing techbriefs configs; mirror
  `nemo.toml`→`nemo_specs.toml`).

## Change 1 — `_nemo.py` relevance filter → Ollama

- Add module helper `_is_ollama_model(model)` — colon-rule mirroring `generate-qa.py:171-185`.
- `QAGenerator.__init__` (~164-182): `self.relevance_model = chunk_cfg.get("relevance_model") or
  self.llm.cfg.model`; if Ollama → `self.eval_client = AsyncOpenAI(base_url=
  chunk_cfg.get("ollama_base_url","http://localhost:11434/v1"), api_key="ollama")` (no
  `OPENAI_API_KEY`); else existing OpenAI behavior. Keep mode/`filter_on` gating + ignore-log branch.
- `evaluate_chunks` (~373-377): `model="gpt-4o-mini"` → `model=self.relevance_model`. Parser,
  `RelevanceJudgment`, and `except → score=1.0` graceful fallback unchanged.
- No change to logical-grouping LLM (already Ollama via `aisa/gen/chat_llm.py::_init_ollama_model`,
  `[llm].model` at `_nemo.py:820`).

## Change 2 — `extract_artifacts.py` → Ollama (OpenAI path preserved)

- Imports: add `from ollama import Client as OllamaClient`; `from langextract import factory as
  lx_factory` if `lx.factory` not attribute-reachable.
- Duplicate `is_ollama_model`/`is_gpt`/`is_gemini` from `generate-qa.py:171-185` (single-file
  standalone — no import).
- `LXConfig` (579-589): add `ollama_host="http://localhost:11434"`, `ollama_num_ctx=16384`,
  `ollama_timeout=600`, `ollama_retries=3`.
- `ChunkLevelExtractor`: `__init__` gate `OPENAI_API_KEY`+`OpenAI()` behind `not is_ollama`, else
  `OllamaClient(host=cfg.ollama_host)`. `extract()` Ollama branch =
  `client.chat(model, [system,user], options={temperature,num_ctx}, format=
  ChunkSignals.model_json_schema(), think=False)` → `ChunkSignals.model_validate_json(...)` with
  `ollama_retries`; shared soft topic-count tail (629-640) + `return signals` unchanged.
- `PavementExtractor`: `__init__` gate key guard, `self.api_key=None` for Ollama.
  `_extract_spans` Ollama branch = `lx.extract(..., config=lx.factory.ModelConfig(
  model_id=cfg.model, provider="OllamaLanguageModel", provider_kwargs={model_url, base_url,
  temperature, num_ctx, timeout}), extraction_passes=..., max_char_buffer=..., show_progress=False)`
  — no `api_key`. Post-proc 687-711 untouched.
- Untouched: `main()`, orchestrator + failure-isolation contract, `ChunkSignals`,
  `SPAN_LEVEL_EXAMPLES`, `_resolve_input_dir`.

## Change 3 — config files

- `cfg/nemo_specs.toml` (edit): `[chunking].relevance_concurrency` 8 → 2.
- `extract_artifacts_specs.toml` (new, from `extract_artifacts.toml`): `[paths].input_dir =
  ./data/specs_20260516/doc-chunks_256_random_logical`; `[artifact_extraction]` `model=
  "gpt-oss:20b"`, `extraction_passes=1`, `chunk_concurrency=1`, `ollama_*` keys.
- `generate-qa_specs.toml` (new, from `generate-qa.toml`): `chunk_dir=
  ./data/specs_20260516/doc-chunks_256_random_logical`, `output_dir=
  ./data/specs_20260516/qa-gen`, `model_qa=model_citations="gpt-oss:20b"`.
- `self-check/self-check-qa_specs.toml` (new, from `self-check-qa.toml`): `input_qa_json=
  ../data/specs_20260516/qa-gen/generated-questions.json`, `model="gpt-oss:20b"`,
  outputs `./self-check-output-specs/`, `max_concurrent_questions` → 4.

## Prerequisites

1. User renames `rawdata-pub` → `rawdata-pubs` (24 `PUB242C09_*.md` at `./rawdata-pubs/parsed-specs/`).
2. Ollama up, started with `num_ctx=16384` (deployment standard, local & cluster); `gpt-oss:20b` +
   `nomic-embed-text:latest` present (verified — no pulls).
3. Use `.venv/bin/python`.

## Verification

Smoke test on 1 doc through all 4 stages, asserting: stage-1 `*-logic-ctx.json` + `*-relevance.json`
sane; stage-2 `errors.{span,chunk}` null, `chunk_signals` schema-valid, non-empty `extractions`;
stage-3 non-empty Q/A/context; stage-4 per-record `evaluation ∈ {0,0.5,1}`. Then full 24-doc run
(stages are file-cached/idempotent — re-run resumes). Detail in SRS §9.

## Risks

R1 langextract span truncation: provider forces `num_ctx=2048` into request options, overriding the
16384 server default — must pass `num_ctx=16384` explicitly (matches standard Ollama startup,
local+cluster). Native-`chat`/`/v1` paths inherit the server default.
R2 `gpt-oss:20b` reasoning leak — chunk/stage3/4 use `think=False`; span+relevance have graceful
fallbacks, tighten prompts if needed. R3 single-GPU serialization — low concurrency + `timeout=600`.
R4 deep nested schema on 20B — grammar-constrained + retries. R5 grouping output — `format="json"`
+ window-end fallback. Full table in SRS §8.
