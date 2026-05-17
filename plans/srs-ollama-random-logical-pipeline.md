# Software Requirements Specification: Ollama Support for the `random_logical` SDG Pipeline

**Feature:** End-to-end local-Ollama execution of the four-stage `random_logical` synthetic-data
pipeline (`_nemo.py --sdg-logical` → `extract_artifacts.py` → `generate-qa.py` →
`self-check/self-check-qa.py`), producing LLM-evaluated questions with **no OpenAI/Gemini calls**.
**Component:** `nvidia-sdq_custom`
**Version:** 0.3 (implementation findings folded in)
**Status:** Implemented + cluster-portable (FR-8). **Full 4-stage local smoke PASSED** on `gpt-oss:20b`, zero egress (bogus-but-non-empty keys), CH01, downstream bounded to 1 logical chunk: S1 202s (R5/R8 ✓) → S2 415s (span 23/5-classes, chunk 5 topics+summary) → S3 532s (15 Q/A/ctx+citations) → S4 157s (15 evals, dist {1.0:14, 0.0:1}). Deliverable `self-check-qa-results.json` produced. `_specs` configs target `gpt-oss:120b`; endpoint resolves from `$OLLAMA_HOST`/`--host`. **Full 24-doc, uncapped run → cluster** (local 20b too slow — §8 R7).
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
  (standard for both local and cluster), so **three** Ollama paths inherit 16384 automatically —
  the native-`chat` chunk path, the `/v1` relevance path, and the logical-grouping `ChatOllama`
  path (`langchain_ollama` injects no 2048 default; that path also has no reasoning-suppression
  knob — R5) — but the explicit `provider_kwargs.num_ctx=16384` on the
  langextract span path is **still mandatory** (it counteracts langextract's 2048 injection).
- Ollama exposes an OpenAI-compatible API at `…/v1`; `openai.AsyncOpenAI(base_url, api_key="ollama")`
  works ⇒ minimal-diff relevance refactor preserving the existing `<json>`/`<scratchpad>` parser.
- Local Ollama has `gpt-oss:20b` and `nomic-embed-text:latest` (no pulls needed).
- `aisa/gen/providers.py:36-41,109-117` builds OpenAI+Google `CHAT_MODELS`/`EMBED_MODELS` at import
  and raises `ValueError: Missing {KEY}` if `OPENAI_API_KEY`/`GOOGLE_API_KEY` are empty ⇒ **non-empty
  keys are required to import `_nemo.py`** (it imports `aisa/gen`) even in all-Ollama mode (never
  used for Ollama network calls). The standalone `extract_artifacts.py`/`generate-qa.py` do **not**
  import `aisa/` and need no OpenAI/Gemini key in Ollama mode. Drives §7.2, §9 zero-egress proof.
- `[llm].request_timeout` is **not** forwarded to `ChatOllama` (`aisa/gen/chat_llm.py:61-73`), so
  `nemo_specs.toml`'s `request_timeout=2` is inert for the grouping path (no timeout risk, no action;
  do not "fix" it into a real 2 s cap). That code path also has no reasoning-suppression knob (R5).

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
  `AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")` (literal — the
  `ollama_base_url` config knob is dropped as scope creep; single local host). The relevance
  *client* does not use `OPENAI_API_KEY`, **but** `aisa/gen/providers.py` still structurally
  requires non-empty `OPENAI_API_KEY`/`GOOGLE_API_KEY` to import `_nemo.py` (§7.2) — the earlier
  "SHALL NOT require OPENAI_API_KEY" framing was wrong. Otherwise retain the current OpenAI
  construction. The existing method-gating and "relevance_filter ignored" log branch SHALL be
  preserved.
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
- **FR-3.2** (revised after smoke — the original `config=ModelConfig(provider=…)`
  approach is **superseded**). Root cause: stock langextract's Ollama provider
  always forces `payload['format']` (json/yaml grammar) on `/api/generate` with
  no thinking suppression — `infer()` derives it from `format_type` (default
  JSON) and `_ollama_query` re-derives it — so gpt-oss (Harmony reasoning) emits
  chain-of-thought and **0 spans parse** (proven by ablation + first smoke).
  `_extract_spans` Ollama branch SHALL instead construct a minimal
  `OllamaLanguageModel` subclass `_OllamaSpanLM` that removes the forced format:
  `__init__` sets `self.format_type=None`; `infer()` overridden to call
  `_ollama_query(..., structured_output_format=None, ...)`. It SHALL call
  `lx.extract(..., model=_OllamaSpanLM(model_id=cfg.model, base_url=cfg.ollama_host,
  temperature=cfg.temperature, num_ctx=cfg.ollama_num_ctx, timeout=cfg.ollama_timeout),
  extraction_passes=…, max_char_buffer=…, show_progress=False)` — **no** `api_key`,
  **no** `config=`/`provider=`. `lx.extract(model=…)` takes precedence over all
  other params and skips schema re-application, so the subclass is used as-is;
  the prompt+`SPAN_LEVEL_EXAMPLES` elicit JSON that langextract's resolver parses
  (verified: 7 valid spans vs 0). The span post-processing
  (bucketing/`SPAN_LEVEL_CLASSES` gating) SHALL be unchanged.
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

### FR-7 — Implementation invariants (from review)
- **FR-7.1** The `LXConfig` field additions (FR-2.2) MUST land in the **same change** as
  `extract_artifacts_specs.toml`'s `ollama_*` keys. `main()` does `LXConfig(**cfg.get("artifact_extraction", {}))`
  (`extract_artifacts.py:828`); an `ollama_*` key without a matching dataclass field aborts with
  `TypeError` before any extraction.
- **FR-7.2** The duplicated `is_ollama_model`/`is_gpt`/`is_gemini` trio in `extract_artifacts.py`
  MUST stay byte-identical to `generate-qa.py:171-185` (add a code comment to that effect); the two
  copies drifting silently desyncs stage-2 from stage-3 routing.
- **FR-7.3** Regression invariant: default `LXConfig.model = "gpt-4o-mini"` (no colon, starts
  "gpt") keeps the OpenAI branch; only a colon-bearing model id flips to Ollama. This is what keeps
  the existing `extract_artifacts.toml`/techbriefs runs on OpenAI — do not change the default.

### FR-8 — Cluster endpoint portability (added post-smoke)
The cluster slurm jobs run Ollama at `OLLAMA_HOST=http://127.0.0.1:$PORT` (dynamic
port) and pass `--host localhost --port $PORT` to scripts. To make the same
`_specs` configs work local **and** cluster:
- **FR-8.1** `extract_artifacts.py` SHALL add `--host`/`--port` CLI and a
  `_resolve_ollama_host()` with precedence **CLI > `[artifact_extraction].ollama_host`
  > `$OLLAMA_HOST` > `http://localhost:11434`** (scheme auto-prepended). `LXConfig.ollama_host`
  default → `None`; `main(..., ollama_host=)` resolves it once and both the chunk
  `OllamaClient` and `_OllamaSpanLM` use the resolved value.
- **FR-8.2** `_nemo.py` relevance client SHALL derive its base URL from
  `$OLLAMA_HOST` (else `http://localhost:11434`) + `/v1` (grouping `ChatOllama`
  and `ollama_api` already honor `$OLLAMA_HOST` via the `ollama` lib).
- **FR-8.3** `extract_artifacts_specs.toml` SHALL **omit** `ollama_host` so it
  resolves from `$OLLAMA_HOST` on the cluster and localhost otherwise.
- **FR-8.4** The four `_specs` configs (`cfg/nemo_specs.toml` `[llm].model`,
  `extract_artifacts_specs.toml`, `generate-qa_specs.toml`,
  `self-check/self-check-qa_specs.toml`) target `gpt-oss:120b` (cluster), with
  the `:20b` local-debug alt in a trailing `#`-comment, and cluster-tuned
  concurrency (`chunk_concurrency=8`, `max_concurrent_questions=16` vs
  `OLLAMA_NUM_PARALLEL=16`). **Slurm jobs for stages 1–2 are now specified** as a single
  combined job in `plans/srs-cluster-pipeline-slurm.md`, which follows the **self-check
  template's `OLLAMA_CONTEXT_LENGTH=65536`/`NUM_PARALLEL=8`** (authoritative for the combined
  specs run; supersedes the generic `32768/16` in §7 item 6 below).

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

New `_specs` configs MUST be created by copying the existing sibling toml **verbatim** and changing
only the rows below. The scripts read several other required keys via direct subscript
(`output_file`, `output_csv_file`, `question_generate_prompt`, `extract_citation_prompt`,
`prompt_eval`, `intermediate_json`, …) — omitting them raises `KeyError` at startup.

| File | Key | Value |
|---|---|---|
| `cfg/nemo_specs.toml` | `[general].data_dir` | `./rawdata-pubs/parsed-specs` (rename already applied — verify 24 files) |
| | `[general].output_dir` | `./data/specs_20260516` |
| | `[llm].model` | `gpt-oss:20b` |
| | `[chunking].method` | `random_logical` |
| | `[chunking].relevance_filter` | `true` |
| | `[chunking].relevance_concurrency` | `2` (was 8) |
| | `[chunking].relevance_model` *(opt.)* | defaults to `[llm].model` (kept; `ollama_base_url` knob dropped) |
| `extract_artifacts_specs.toml` | `[paths].input_dir` | `./data/specs_20260516/doc-chunks_256_random_logical` |
| | `[artifact_extraction].model` | `gpt-oss:20b` |
| | `…extraction_passes` | `1` |
| | `…chunk_concurrency` | `1` |
| | `…ollama_num_ctx` / `ollama_timeout` / `ollama_retries` | `16384` (matches standard Ollama startup, local+cluster) / `600` / `3` |
| `generate-qa_specs.toml` | `chunk_dir` | `./data/specs_20260516/doc-chunks_256_random_logical` |
| | `output_dir` | `./data/specs_20260516/qa-gen` |
| | `model_qa`, `model_citations` | `gpt-oss:20b` |
| `self-check/self-check-qa_specs.toml` | `input_qa_json` | `../data/specs_20260516/qa-gen/generated-questions.json` — resolved relative to `self-check/` (stage-4 runs with CWD there); MUST be the WITH-context file, not `_wo_context` (self-check needs `context_text`) |
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

1. **Corpus present (verify — already done):** `./rawdata-pubs/parsed-specs/*.md` resolves to the
   24 `PUB242C09_*.md` (the `rawdata-pub`→`rawdata-pubs` rename has been applied; singular
   `rawdata-pub` no longer exists). If absent, `_nemo.py` globs nothing and **returns exit 0 silently**
   — §9 makes the 24-file count a hard gate, not a step.
2. **API keys structurally required at import (even all-Ollama).** `aisa/gen/providers.py:36-41,
   109-117` builds OpenAI+Google `CHAT_MODELS`/`EMBED_MODELS` at import; `BaseInfo` raises if
   `OPENAI_API_KEY`/`GOOGLE_API_KEY` empty. So **`_nemo.py` (stage 1) only** needs **non-empty**
   `OPENAI_API_KEY` and `GOOGLE_API_KEY` in env/`.env` to *import* (it imports `aisa/gen`); values
   are **never used for network calls** under Ollama. The standalone `extract_artifacts.py`/
   `generate-qa.py` (no `aisa/` import) need no OpenAI/Gemini key in Ollama mode. Blanking the keys
   breaks the stage-1 import, so the §9 zero-egress proof uses *bogus-but-non-empty* keys, not
   blank. `.env` currently has live keys.
3. **Ollama** running, started with `num_ctx=16384` (deployment standard, local & cluster);
   `gpt-oss:20b` + `nomic-embed-text:latest` present (verified — no pulls). Must stay reachable for
   the whole run: `_nemo.py main()` constructs `BaseLLM` (→ `check_existing_model("gpt-oss:20b")` →
   `exit()` if absent) and `Embedder` (→ `ollama_api.list_models()` → `exit()` if Ollama down) even
   though `--sdg-logical` never embeds; `nomic-embed-text:latest` must appear in `ollama list` so
   the runtime `Embedder` routes to the Ollama provider instead of attempting a HuggingFace load.
4. **Import-time HF model.** `_nemo.py:130` default-arg `Embedder(EmbedConfig())` is evaluated at
   class-body execution → builds `HuggingFaceEmbeddings("sentence-transformers/all-MiniLM-L6-v2")`
   (HF construction in `aisa/gen/embed.py:42`; default model string `embed.py:21`);
   that model must be in the local HF cache or network-reachable at import (libs installed per
   `reqs.txt`; only the weights are the open risk).
5. Interpreter: `.venv/bin/python`. Native-`chat`/`/v1` paths inherit the 16384 server default;
   the langextract span path requires the explicit `num_ctx` override (§2, §8 R1).
6. **Cluster:** slurm exports `OLLAMA_HOST=http://127.0.0.1:$PORT` (dynamic). The
   `OLLAMA_CONTEXT_LENGTH`/`NUM_PARALLEL` values differ by template (`generate-qa.slurm`
   `32768/16`; `self-check-qa.slurm` `65536/8`); the combined specs job
   (`plans/srs-cluster-pipeline-slurm.md`) follows **`65536/8`** (see its §8 R9 VRAM caveat).
   Per FR-8 the configs/code resolve the endpoint from `$OLLAMA_HOST` (or `--host/--port`), so no
   per-job port edit is needed. Local: no `$OLLAMA_HOST` → `http://localhost:11434`.
7. Disk: negligible — per-doc intermediates (`-chunks`/`-logic-chunks`/`-relevance`/`-logic-ctx`/
   `-logic-artifacts`) + QA + self-check JSON for 24 docs total ~tens of MB (negligible — not a
   tracked risk).

---

## 8. Risks & Mitigations

| # | Risk | Mitigation |
|---|---|---|
| R1 | langextract span truncation: provider injects `num_ctx=2048` into request `options`, **overriding the 16384 server default** ⇒ span prompt + 3 large examples silently truncated, empty extractions | Always pass `num_ctx=16384` via `provider_kwargs` (mandatory despite the server default); native-`chat`/`/v1` inherit 16384; §9 asserts non-empty `extractions` |
| R2 | gpt-oss (Harmony reasoning — **20b AND 120b**) CoT pollution. **Materialized in smoke:** langextract's stock Ollama provider hardcodes `payload['format']` on `/api/generate` with no thinking suppression ⇒ gpt-oss emits CoT, **0 spans parsed**. Prompt-tightening proven *insufficient* (grammar-level, not prompt-level). Architectural & size-independent — same on cluster 120b. | **Resolved (FR-3.2):** `_OllamaSpanLM` subclass strips the forced `format`; clean JSON → 7 valid spans (ablation + smoke verified). Chunk-level + stages 3/4 + relevance use native `think=False`/tolerant parsers and were unaffected. |
| R3 | Single local GPU serializes calls ⇒ latency/timeouts | `chunk_concurrency=1`, `extraction_passes=1`, `relevance_concurrency=2`, `max_concurrent_questions=4`, `ollama_timeout=600` |
| R4 | Deep nested `ChunkSignals` schema on 20B (adherence + latency) | Ollama `format=<schema>` grammar-constrains output (slow on 20B — latency folds into R7); `ollama_retries` + soft topic clamp; flatten-schema only if repeated `$ref` failures |
| R5 | **Silent grouping collapse:** `request_timeout=2` is inert for `ChatOllama` (not forwarded — confirmed), but reasoning pollution can make `_validate_split_response` return `[]` for every window ⇒ logical chunks silently collapse to recursive windows while files still write | `ChatOllama(format="json")` + window-end fallback prevents a crash; **§9 asserts logical-chunk count ≠ raw recursive-piece count** and greps for fallback log lines so total collapse *fails* acceptance |
| R6 | langextract provider wiring | **Resolved/obsoleted:** span path no longer uses `provider=`/`config=` resolution — FR-3.2 passes a constructed `_OllamaSpanLM` via `lx.extract(model=…)`. The `resolve_provider` lazy-registry caveat is moot. Verified end-to-end in §9 smoke. |
| R7 | **Unbounded wall-clock** confirmed materially severe. **Measured (local `gpt-oss:20b`, 1 doc CH01 ≈ 8 KB → 9 recursive pieces / 4 logical chunks):** stage 1 ≈ **203 s**; stage 2 ≈ **1400 s** (~23 min, span 1-pass + chunk schema ×4); stage 3 not completed locally (12 bounded QA tasks alone projected ≥30 min) — user stopped local run to use the cluster. Largest chapters are ~13× CH01 ⇒ full 24-doc on local 20b ≈ many hours/overnight (confirmed). | Cluster (`gpt-oss:120b`, real GPU) is the intended venue for the full run; per-stage numbers above are the local-20b baseline. Re-measure stage 1–4 per-doc on the cluster before the 24-doc launch so a long run isn't mistaken for a hang. |
| R8 | **Silent keep-all relevance:** `evaluate_chunks` per-call `except→score=1.0` + `path2chunks` `except→keep-all` ⇒ a broken Ollama relevance path yields a green run with the filter silently off (defeats the reason to refactor vs. disable) | §9 asserts `-relevance.json` has ≥1 `score != 1.0` **and** the run log has no `relevance filter failed` / `defaulting to score=1.0` |
| R9 | Resume granularity is **per-doc**, not per-context; a kill/timeout mid-doc re-does that doc; `overwrite=False` is the only mode (no CLI flag) | Documented expectation; to redo a degraded doc, delete its `-logic-*`/`-artifacts` file — cache does not self-heal |
| R10 | Non-determinism: `temperature=0` but Ollama/`gpt-oss:20b` not bit-deterministic ⇒ questions/scores vary run-to-run | Acceptance is shape/existence-based (not value-based) by design; stated so a differing re-run is not treated as a regression |

---

## 9. Acceptance Criteria / Test Plan

**Hard gate (must pass or stop):** `ls rawdata-pubs/parsed-specs/*.md | wc -l` == 24. Confirm CLI
flags via `-h`: `extract_artifacts.py --cfg`, `generate-qa.py --config`, `self-check-qa.py --config`.

**Zero-egress proof (mandatory invocation condition).** Do **not** blank the keys — that breaks
import (§7.2). Run every smoke stage with **bogus but non-empty** keys:
`env OPENAI_API_KEY=sk-bogus-zero-egress GOOGLE_API_KEY=bogus .venv/bin/python …`. The import gate
passes; any *actual* OpenAI/Gemini call fails loudly (401/invalid) instead of silently billing the
real `.env` keys; Ollama paths are unaffected. **Success under bogus keys IS the zero-egress proof.**

**1-doc subset (no `--limit` exists for stages 1–3).** Isolate one doc and override input via CLI
(no toml edit): `mkdir -p /tmp/specs-smoke && cp rawdata-pubs/parsed-specs/<one>.md /tmp/specs-smoke/`
then run stage 1 with `--input_dir /tmp/specs-smoke` (`_nemo.py:847-848` override). Keep
`output_dir = ./data/specs_20260516` so stages 2–4 (which glob the chunk_dir) process just that doc
and the later full run reuses the cache. **Precondition:**
`./data/specs_20260516/doc-chunks_256_random_logical/` must be empty/absent before the smoke run,
else stage-2 globs any pre-existing `*-logic-ctx.json` too.

**Smoke test (1 doc; wrap each stage with wall-clock — `/usr/bin/time -l` or `date` bracketing):**
1. stage 1 (`_nemo.py --sdg-logical --cfg cfg/nemo_specs.toml --input_dir /tmp/specs-smoke`) →
   `{doc}-logic-ctx.json` exists; **logical-chunk count ≠ recursive-piece count** in
   `{doc}-chunks.json` (grouping ran — R5); `{doc}-relevance.json` has ≥1 `score != 1.0` and the log
   has **no** `relevance filter failed` / `defaulting to score=1.0` (filter ran on Ollama — R8).
2. stage 2 (`extract_artifacts.py --cfg extract_artifacts_specs.toml`) → `{doc}-logic-artifacts.json`
   with `errors.chunk` null + schema-valid `chunk_signals`; `errors.span` null + **non-empty**
   `extractions` (empty ⇒ R1/R2 — raise `ollama_num_ctx`/tighten prompt, then **delete the file**
   before re-run — cache won't self-heal, R9).
3. stage 3 (`generate-qa.py --config generate-qa_specs.toml`) → `generated-questions.json` with
   non-empty Q/A/`context_text`.
4. `cd self-check && self-check-qa.py --config ./self-check-qa_specs.toml` →
   `self-check-output-specs/self-check-qa-results.json`, per-record `evaluation ∈ {0,0.5,1}`.

**Acceptance:** all four stages complete **under bogus keys** (zero OpenAI/Gemini egress); the R5 and
R8 assertions pass; deliverable JSON has evaluated questions for the smoke doc. Record per-stage
seconds and **publish the 24-doc projection in §8 R7** before launching the full run.

**Full run:** stage 1 *without* `--input_dir` (re-globs all 24; the smoke doc is a cache-hit via
`-logic-chunks.json`/`-logic-ctx.json` existence); stages 2–4 re-run and resume per-doc.

---

## 10. Future Work (out of scope)

- Cluster/slurm scaling with a larger judge (`gpt-oss:120b`) per `self-check/cluster/`.
- `aisa/gen/chat_llm.py` explicit reasoning suppression for `ChatOllama` if R5 materializes.
- `langextract` schema-constrained Ollama (if upstream adds it) to harden span-level.
- Wiring `filter-questions-citation-eval.py` + `eval2.py` for retrieval benchmarking.
