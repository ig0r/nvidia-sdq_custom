# Software Requirements Specification: Ollama Support for the `random_logical` SDG Pipeline

**Feature:** End-to-end local-Ollama execution of the four-stage `random_logical` synthetic-data
pipeline (`_nemo.py --sdg-logical` → `extract_artifacts.py` → `generate-qa.py` →
`self-check/self-check-qa.py`), producing LLM-evaluated questions with **no OpenAI/Gemini calls**.
**Component:** `nvidia-sdq_custom`
**Version:** 0.1 (draft)
**Status:** Proposed
**Companion plan:** `plans/plan-ollama-random-logical-pipeline.md`

---

## 1. Introduction

### 1.1 Purpose
This SRS defines the requirements to run the existing `random_logical` SDG pipeline entirely on a
local Ollama server (`gpt-oss:20b`, `http://localhost:11434`) for the PennDOT spec corpus, yielding
a set of self-checked (LLM-as-judge) question/answer records. It specifies two source-code
refactors (the only OpenAI-bound paths) and a set of stage-scoped config files; stages 3–4 are
already Ollama-capable and require configuration only.

### 1.2 Scope

**In scope**
- Refactor `_nemo.py` relevance filter (`QAGenerator.__init__`, `evaluate_chunks`) to route the
  per-piece relevance evaluation to Ollama via Ollama's OpenAI-compatible endpoint, model
  config-driven. OpenAI behavior preserved when a non-Ollama model is configured.
- Refactor `extract_artifacts.py` (`LXConfig`, `ChunkLevelExtractor`, `PavementExtractor`) to add an
  Ollama path: chunk-level via the `ollama` client (`format=<json schema>`, `think=False`);
  span-level via `langextract`'s Ollama provider. OpenAI path preserved.
- New module helper `_is_ollama_model` in `_nemo.py`; duplicated `is_ollama_model`/`is_gpt`/
  `is_gemini` trio in `extract_artifacts.py`.
- Four stage-scoped configs: edit `cfg/nemo_specs.toml`; create `extract_artifacts_specs.toml`,
  `generate-qa_specs.toml`, `self-check/self-check-qa_specs.toml`.
- A 1-doc smoke test and full 24-doc run procedure.

**Out of scope**
- Code changes to `generate-qa.py` and `self-check/self-check-qa.py` (already Ollama-native;
  config-only).
- Changes to `aisa/` (the logical-grouping LLM already routes to Ollama via
  `aisa/gen/chat_llm.py::_init_ollama_model`).
- Retrieval-eval scripts (`eval.py`, `eval2.py`, `filter-questions-citation-eval.py`) — not on the
  path to evaluated questions.
- Schema redesign of `ChunkSignals` (contingency only; see §8 R4).
- Renaming the input directory (user action; see §7).

---

## 2. Background / Current State

The `random_logical` pipeline produces, per `.md` doc, recursive pieces → LLM-grouped logical
chunks → 1:1 logical contexts → span+chunk artifacts → questions → self-check scores. Stage I/O is
file-based and idempotent (each stage skips when its output exists and `overwrite` is false).

OpenAI bindings found by audit:

| Locus | Binding | Disposition |
|---|---|---|
| `_nemo.py::QAGenerator.__init__` ~164-182 | eager `AsyncOpenAI(api_key=…)`; raises without `OPENAI_API_KEY` when `relevance_filter` on | Refactor (FR-1) |
| `_nemo.py::evaluate_chunks` ~373-377 | `self.eval_client.chat.completions.create(model="gpt-4o-mini", …)` | Refactor (FR-1) |
| `extract_artifacts.py::ChunkLevelExtractor` 598-641 | `OpenAI(...)` + `beta.chat.completions.parse(response_format=ChunkSignals)` | Refactor (FR-2) |
| `extract_artifacts.py::PavementExtractor._extract_spans` 676-686 | `lx.extract(model_id=…, api_key=…)` (OpenAI) | Refactor (FR-3) |
| `extract_artifacts.py` `__init__` guards ~600-604, ~653-657 | raise without `OPENAI_API_KEY` | Relax for Ollama (FR-2/FR-3) |
| `generate-qa.py` | already routes via `is_ollama_model`; `query_ollama_structured` (`think=False`, `format=schema`) | Config only (FR-4) |
| `self-check/self-check-qa.py` | already Ollama-only (`ollama.AsyncClient`, `format=schema`) | Config only (FR-5) |
| `aisa/gen/chat_llm.py::_init_ollama_model` | `ChatOllama(format="json" if json_mode)` | No change |

Verified environment facts:
- `langextract` 1.2.1: Ollama path is **JSON-mode only** (not schema-constrained), default
  `num_ctx=2048`, **no `think=False`**; auto-routes `^gpt-oss` model ids to Ollama but does **not**
  auto-inject the Ollama URL for non-"ollama" ids. ⇒ chunk-level must use the `ollama` client
  directly; span-level must pass an explicit large `num_ctx` and provider.
- langextract sends its `num_ctx=2048` in the request `options`, which **overrides the server's
  context length** (`OLLAMA_CONTEXT_LENGTH`). This deployment starts Ollama with `num_ctx=16384`
  (standard for both local and cluster), so the native-`chat` chunk path and the `/v1` relevance
  path inherit 16384 automatically — but the explicit `provider_kwargs.num_ctx=16384` on the
  langextract span path is **still mandatory** (it counteracts langextract's 2048 injection).
- Ollama exposes an OpenAI-compatible API at `…/v1`; `openai.AsyncOpenAI(base_url, api_key="ollama")`
  works ⇒ minimal-diff relevance refactor preserving the existing `<json>`/`<scratchpad>` parser.
- Local Ollama has `gpt-oss:20b` and `nomic-embed-text:latest` (no pulls needed).

---

## 3. Functional Requirements

### FR-1 — `_nemo.py` relevance filter on Ollama
- **FR-1.1** Add `_is_ollama_model(model: str) -> bool` (module scope): `True` if `":" in model`,
  else `not (model.startswith("gpt") or model.startswith("gemini"))`. Mirrors
  `generate-qa.py:171-185`.
- **FR-1.2** `QAGenerator.__init__` SHALL compute
  `self.relevance_model = chunk_cfg.get("relevance_model") or self.llm.cfg.model`.
- **FR-1.3** When `relevance_filter` is on and `method == "random_logical"`: if
  `_is_ollama_model(self.relevance_model)`, construct
  `AsyncOpenAI(base_url=chunk_cfg.get("ollama_base_url","http://localhost:11434/v1"),
  api_key="ollama")` and SHALL NOT require `OPENAI_API_KEY`; otherwise retain the current OpenAI
  construction (require `OPENAI_API_KEY`). The existing method-gating and "relevance_filter
  ignored" log branch SHALL be preserved.
- **FR-1.4** `evaluate_chunks` SHALL call the model via `model=self.relevance_model` (replacing the
  hardcoded `"gpt-4o-mini"`). The `_JSON_BLOCK_RE`/`_SCRATCHPAD_BLOCK_RE` parsing,
  `RelevanceJudgment` validation, per-piece concurrency, on-disk `-relevance.json` caching, and the
  existing `except → score=1.0` (keep-all) fallback SHALL be unchanged.
- **FR-1.5** No change to `path2chunks`, `group_kept_pieces`, the chunker, or the grouping LLM.

### FR-2 — `extract_artifacts.py` chunk-level on Ollama
- **FR-2.1** Add `from ollama import Client as OllamaClient`; duplicate `is_ollama_model`/`is_gpt`/
  `is_gemini` (no cross-file import — module is single-file standalone).
- **FR-2.2** `LXConfig` SHALL gain `ollama_host: str = "http://localhost:11434"`,
  `ollama_num_ctx: int = 16384`, `ollama_timeout: int = 600`, `ollama_retries: int = 3`. Existing
  OpenAI tomls SHALL continue to load unchanged (extra keys default).
- **FR-2.3** `ChunkLevelExtractor.__init__` SHALL select provider via `is_ollama_model(cfg.model)`;
  the `OPENAI_API_KEY` guard and `OpenAI(...)` client SHALL be gated to the non-Ollama branch; the
  Ollama branch SHALL build `OllamaClient(host=cfg.ollama_host)`. Prompt loading unchanged.
- **FR-2.4** `ChunkLevelExtractor.extract` Ollama branch SHALL call
  `client.chat(model=cfg.model, messages=[system,user],
  options={"temperature":cfg.temperature,"num_ctx":cfg.ollama_num_ctx},
  format=ChunkSignals.model_json_schema(), think=False)`, validate via
  `ChunkSignals.model_validate_json(resp["message"]["content"])`, retry up to `cfg.ollama_retries`,
  and **raise** on exhaustion (preserving the orchestrator's raise-on-failure contract). The
  existing soft topic-count validation (0/>5) and `return signals` SHALL be the shared tail for
  both providers.

### FR-3 — `extract_artifacts.py` span-level on Ollama (`langextract`)
- **FR-3.1** `PavementExtractor.__init__` SHALL select provider via `is_ollama_model(cfg.model)`;
  the `OPENAI_API_KEY` guard SHALL be gated to non-Ollama; `self.api_key` SHALL be `None` for
  Ollama. `ChunkLevelExtractor(cfg)` construction unchanged (self-routes).
- **FR-3.2** `_extract_spans` Ollama branch SHALL call `lx.extract` with
  `config=lx.factory.ModelConfig(model_id=cfg.model, provider="OllamaLanguageModel",
  provider_kwargs={"model_url":cfg.ollama_host,"base_url":cfg.ollama_host,
  "temperature":cfg.temperature,"num_ctx":cfg.ollama_num_ctx,"timeout":cfg.ollama_timeout})`,
  passing `extraction_passes`, `max_char_buffer`, `show_progress=False`, and **no** `api_key`. The
  span post-processing (bucketing/`SPAN_LEVEL_CLASSES` gating, lines 687-711) SHALL be unchanged.
- **FR-3.3** The OpenAI branch SHALL remain the pre-existing call verbatim (incl.
  `api_key=self.api_key`). The 2-worker span/chunk `ThreadPoolExecutor` and per-call
  failure-isolation (`errors.{span,chunk}`) SHALL be unchanged.

### FR-4 — `generate-qa.py` (config only)
- **FR-4.1** A `generate-qa_specs.toml` SHALL set `chunk_dir`/`output_dir` to the specs paths and
  `model_qa = model_citations = "gpt-oss:20b"`. No code change. Provider routing
  (`is_ollama_model`), `query_ollama_structured` (`think=False`, `format=schema`), and
  `--host/--port` defaults (localhost:11434) are reused as-is.

### FR-5 — `self-check/self-check-qa.py` (config only)
- **FR-5.1** A `self-check/self-check-qa_specs.toml` SHALL set `input_qa_json` to the stage-3
  output, `model = "gpt-oss:20b"`, outputs under `./self-check-output-specs/`, and a
  GPU-appropriate `max_concurrent_questions`. No code change.

### FR-6 — Configuration set
- **FR-6.1** `cfg/nemo_specs.toml` (exists; `gpt-oss:20b`, `output_dir=./data/specs_20260516`,
  `method=random_logical`, `relevance_filter=true`, `data_dir=./rawdata-pubs/parsed-specs`) SHALL be
  edited to set `[chunking].relevance_concurrency = 2`.
- **FR-6.2** New configs SHALL be `_specs`-suffixed copies (existing techbriefs configs untouched).
- **FR-6.3** `extract_artifacts_specs.toml` SHALL use the `[paths]` + `[artifact_extraction]` shape
  (NOT the `cfg/nemo.toml` shape — `main()` raises `KeyError` if `[langextract]` present without
  `[artifact_extraction]`), with `input_dir = ./data/specs_20260516/doc-chunks_256_random_logical`,
  `model = "gpt-oss:20b"`, `extraction_passes = 1`, `chunk_concurrency = 1`, `ollama_*` keys.

---

## 4. External Interfaces

- **Ollama native client** (`ollama.Client.chat`): stage-2 chunk-level, stage-3, stage-4 —
  `format=<pydantic json schema>`, `think=False`, `options.num_ctx`, `options.temperature`.
- **Ollama OpenAI-compatible** (`AsyncOpenAI(base_url="…/v1", api_key="ollama")`): stage-1 relevance
  — `chat.completions.create(model=…, messages=…, temperature=0)`; response parsed via existing
  `<json>`/`<scratchpad>` regex + `RelevanceJudgment`.
- **`langextract` Ollama provider** (`lx.factory.ModelConfig(provider="OllamaLanguageModel", …)`):
  stage-2 span-level — JSON-mode, explicit `num_ctx`/`timeout`/`model_url`.
- **Ollama list** (`aisa/gen/ollama_api.list_models`): called at `_nemo.py` import via the
  `Embedder` default; Ollama MUST be reachable or the process `exit()`s.

---

## 5. Configuration Schema (effective values)

| File | Key | Value |
|---|---|---|
| `cfg/nemo_specs.toml` | `[general].data_dir` | `./rawdata-pubs/parsed-specs` (unchanged; user renames dir) |
| | `[general].output_dir` | `./data/specs_20260516` |
| | `[llm].model` | `gpt-oss:20b` |
| | `[chunking].method` | `random_logical` |
| | `[chunking].relevance_filter` | `true` |
| | `[chunking].relevance_concurrency` | `2` (was 8) |
| | `[chunking].relevance_model` *(opt.)* | defaults to `[llm].model` |
| | `[chunking].ollama_base_url` *(opt.)* | `http://localhost:11434/v1` |
| `extract_artifacts_specs.toml` | `[paths].input_dir` | `./data/specs_20260516/doc-chunks_256_random_logical` |
| | `[artifact_extraction].model` | `gpt-oss:20b` |
| | `…extraction_passes` | `1` |
| | `…chunk_concurrency` | `1` |
| | `…ollama_num_ctx` / `ollama_timeout` / `ollama_retries` | `16384` (matches standard Ollama startup, local+cluster) / `600` / `3` |
| `generate-qa_specs.toml` | `chunk_dir` | `./data/specs_20260516/doc-chunks_256_random_logical` |
| | `output_dir` | `./data/specs_20260516/qa-gen` |
| | `model_qa`, `model_citations` | `gpt-oss:20b` |
| `self-check/self-check-qa_specs.toml` | `input_qa_json` | `../data/specs_20260516/qa-gen/generated-questions.json` |
| | `model` | `gpt-oss:20b` |
| | `output_*` | `./self-check-output-specs/…` |
| | `max_concurrent_questions` | `4` |

---

## 6. Data Flow / Artifacts

`./data/specs_20260516/`
- `doc-chunks_256_random_logical/{doc}-chunks.json` (recursive pieces)
- `…/{doc}-logic-chunks.json` (LLM-grouped; stage-1 cache key)
- `…/{doc}-relevance.json` (per-piece 0/0.5/1; stage-1 when filter on)
- `…/{doc}-logic-ctx.json` (1:1 contexts; stage-1 `_build_logical_contexts`)
- `…/{doc}-logic-artifacts.json` (stage-2; span `extractions` + `chunk_signals` + `errors`)
- `qa-gen/generated-questions.json` (+ `_qa_only`, `_wo_context`, `.csv`) (stage-3)
- `self-check/self-check-output-specs/self-check-qa-results.json` (+ `_wo_context`, `.csv`)
  — **deliverable**: each record `evaluation ∈ {0, 0.5, 1}` (`-1` = no context, skipped).

Stage `_nemo.py --sdg-logical` runs `path2chunks` then `_build_logical_contexts` in one invocation.

---

## 7. Prerequisites / Assumptions

1. User renames `rawdata-pub` → `rawdata-pubs` so `./rawdata-pubs/parsed-specs/` holds the 24
   `PUB242C09_*.md`. (As-is, `_nemo.py` finds no `.md` and returns.)
2. Ollama server running; `gpt-oss:20b` and `nomic-embed-text:latest` pulled (verified present).
   Ollama must stay reachable for the whole run (`_nemo.py` import constructs `Embedder` →
   `ollama_api.list_models()` → `exit()` if down, even though `--sdg-logical` never embeds).
   Ollama is started with context length **16384** (the deployment standard, local & cluster); the
   explicit `num_ctx` values in §5 match this. Native-`chat` and `/v1` relevance paths inherit the
   server default; the langextract span path requires the explicit override (see §2, §8 R1).
3. Interpreter: `.venv/bin/python`.

---

## 8. Risks & Mitigations

| # | Risk | Mitigation |
|---|---|---|
| R1 | langextract span truncation: its provider injects `num_ctx=2048` into request `options`, **overriding the 16384 server default** ⇒ span prompt + 3 large examples silently truncated, empty extractions | Always pass `num_ctx=16384` via `provider_kwargs` (mandatory despite the 16384 server default — counteracts langextract's 2048 injection); native-`chat`/`/v1` paths already inherit 16384; smoke test asserts non-empty `extractions` |
| R2 | `gpt-oss:20b` reasoning-token pollution | Chunk path + stages 3/4 use `think=False`. Span path (langextract `/api/generate`, no think) and relevance (`/v1`) have graceful fallbacks (tolerant parser; `except → score=1.0`). If smoke test breaks, add "output only JSON, no preamble" to span/relevance prompts |
| R3 | Single local GPU serializes concurrent calls ⇒ latency/timeouts | `chunk_concurrency=1`, `extraction_passes=1`, `relevance_concurrency=2`, `max_concurrent_questions=4`, `ollama_timeout=600` |
| R4 | Deep nested `ChunkSignals` schema on a 20B model | Ollama `format=<schema>` grammar-constrains decoding; `ollama_retries` + existing soft topic clamp. Flatten-schema only as contingency if repeated `$ref` validation failures observed |
| R5 | Stage-1 grouping output (`{"split_after":[…]}`) malformed under reasoning model | `ChatOllama(format="json")` constrains; `_llm_split_decisions` already falls back to window-end splits on bad/empty response (degrade, not crash) |
| R6 | `lx.factory`/provider-name resolution | Validate `lx.factory.ModelConfig` reachability + `provider="OllamaLanguageModel"` resolves (one-line probe before full run) |

---

## 9. Acceptance Criteria / Test Plan

**Pre-flight:** `ls rawdata-pubs/parsed-specs/*.md | wc -l` → 24;
`.venv/bin/python extract_artifacts.py -h` / `generate-qa.py -h` confirm config flags.

**Smoke test (1 doc; stages are idempotent so a single-doc subset suffices):**
1. `.venv/bin/python _nemo.py --sdg-logical --cfg cfg/nemo_specs.toml` → `{doc}-logic-ctx.json`
   exists, sane logical-chunk count; `{doc}-relevance.json` scored 0/0.5/1 (relevance ran on Ollama,
   no OpenAI call).
2. `.venv/bin/python extract_artifacts.py --cfg extract_artifacts_specs.toml` →
   `{doc}-logic-artifacts.json` with `errors.chunk` null + schema-valid `chunk_signals`,
   `errors.span` null + non-empty `extractions` (empty ⇒ R1/R2 — raise `ollama_num_ctx` / tighten
   prompt).
3. `.venv/bin/python generate-qa.py --config generate-qa_specs.toml` → `generated-questions.json`
   with non-empty Q/A/context.
4. `cd self-check && .venv/bin/python self-check-qa.py --config ./self-check-qa_specs.toml` →
   `self-check-output-specs/self-check-qa-results.json` with per-record `evaluation ∈ {0,0.5,1}`.

**Acceptance:** all four stages complete with **zero OpenAI/Gemini network calls**; the deliverable
JSON contains evaluated questions for the smoke doc. **Full run:** repeat without the subset over
all 24 docs (re-running resumes via file cache).

---

## 10. Future Work (out of scope)

- Cluster/slurm scaling with a larger judge (`gpt-oss:120b`) per `self-check/cluster/`.
- `aisa/gen/chat_llm.py` explicit reasoning suppression for `ChatOllama` if R5 materializes.
- `langextract` schema-constrained Ollama (if upstream adds it) to harden span-level.
- Wiring `filter-questions-citation-eval.py` + `eval2.py` for retrieval benchmarking.
