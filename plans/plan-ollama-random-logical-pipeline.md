# Plan: Ollama support for the whole `random_logical` pipeline (local debug, `gpt-oss:20b`)

**Companion SRS:** `plans/srs-ollama-random-logical-pipeline.md`
**Status:** Proposed — revised after 3-agent review (see SRS §8 R7–R10, §9 zero-egress proof, §3 FR-7)
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
  self.llm.cfg.model`; if Ollama → `self.eval_client = AsyncOpenAI(
  base_url="http://localhost:11434/v1", api_key="ollama")` (literal — `ollama_base_url` knob dropped
  per SRS FR-1.3; `OPENAI_API_KEY` still required at import per Prereq 2 / SRS §7.2); else existing
  OpenAI behavior. Keep mode/`filter_on` gating + ignore-log branch.
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
  `_extract_spans` Ollama branch = `lx.extract(..., model=_OllamaSpanLM(model_id, base_url,
  temperature, num_ctx, timeout), extraction_passes, max_char_buffer, show_progress=False)` — no
  `api_key`, no `config=`. `_OllamaSpanLM` = minimal `OllamaLanguageModel` subclass that strips
  langextract's hardcoded request `format` (the gpt-oss-Harmony CoT collision; SRS FR-3.2/R2,
  ablation+smoke-verified). Post-proc untouched.
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

## Change 4 — cluster endpoint portability (SRS FR-8, post-smoke)

- `extract_artifacts.py`: `--host`/`--port` CLI + `_resolve_ollama_host()` (CLI > toml
  `ollama_host` > `$OLLAMA_HOST` > localhost); `LXConfig.ollama_host` default → `None`, resolved in
  `main()`. `_nemo.py` relevance base URL derived from `$OLLAMA_HOST` (+`/v1`).
- `extract_artifacts_specs.toml` omits `ollama_host` (resolves from `$OLLAMA_HOST` on cluster).
- All four `_specs` configs retargeted to `gpt-oss:120b` (`:20b` kept as `#`-comment), cluster
  concurrency (`chunk_concurrency=8`, `max_concurrent_questions=16`). New stage-1/2 slurm jobs:
  out of scope (user-deferred).

## Prerequisites

1. Corpus present (verified — rename already applied): `./rawdata-pubs/parsed-specs/*.md` = 24
   `PUB242C09_*.md`; singular `rawdata-pub` is now empty. Missing ⇒ `_nemo.py` exits 0 silently
   (§9 makes the 24-count a hard gate).
2. **Non-empty** `OPENAI_API_KEY` + `GOOGLE_API_KEY` in env/`.env` — structurally required to
   *import* the pipeline (`aisa/gen/providers.py:36-41,109-117`), never used for Ollama calls.
   Blanking breaks import; zero-egress proof uses bogus-but-non-empty keys (SRS §9).
3. Ollama up, started with `num_ctx=16384` (deployment standard, local & cluster); `gpt-oss:20b` +
   `nomic-embed-text:latest` present (verified — no pulls). Import-time default-arg
   `Embedder(EmbedConfig())` builds HF `all-MiniLM-L6-v2` — must be cached/reachable.
4. Use `.venv/bin/python`.

## Verification

Run the 1-doc smoke test with **bogus-but-non-empty** `OPENAI_API_KEY`/`GOOGLE_API_KEY` (success
under bogus keys = zero OpenAI/Gemini egress proof; blanking breaks import). 1-doc subset via a
throwaway dir + `_nemo.py --input_dir` (no `--limit` exists for stages 1–3); keep `output_dir` so
the full run reuses cache. Assert beyond file-existence: stage-1 logical-chunk count ≠
recursive-piece count **and** `-relevance.json` has ≥1 `score!=1.0` with no fallback log lines
(catches silent collapse/keep-all — SRS R5/R8); stage-2 `errors.{span,chunk}` null + non-empty
`extractions`; stage-3 non-empty Q/A/`context_text`; stage-4 `evaluation ∈ {0,0.5,1}`. Time each
stage and publish the 24-doc projection (SRS R7) before the full run. Detail in SRS §9.

## Risks

R1 langextract span truncation: provider forces `num_ctx=2048` into request options, overriding the
16384 server default — must pass `num_ctx=16384` explicitly (matches standard Ollama startup,
local+cluster). Native-`chat`/`/v1` paths inherit the server default.
R2 `gpt-oss:20b` reasoning leak — chunk/stage3/4 use `think=False`; span+relevance have graceful
fallbacks, tighten prompts if needed. R3 single-GPU serialization — low concurrency + `timeout=600`.
R4 deep nested schema on 20B — grammar-constrained + retries. R5 grouping output — `format="json"`
+ window-end fallback (note `request_timeout=2` is inert for `ChatOllama`). R7 unbounded wall-clock
on one GPU — measure on smoke, project 24-doc before full run. R8 silent keep-all relevance — assert
score discrimination + no fallback logs. Full table (R1–R10) in SRS §8.
