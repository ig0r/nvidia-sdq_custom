# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Custom implementation of NVIDIA's Nemotron Embed Recipe for synthetic data generation (SDG) and data preparation, targeting fine-tuning of embedding models on domain-specific `.md` corpora (pavement-engineering tech briefs and PennDOT specs). Unlike the reference recipe it does not depend on NeMo distributed runtime — chunking/SDG plugs into the project's own `aisa` LLM/embedder abstractions, and the downstream artifact/QA/eval stages are self-contained standalone scripts.

The repo has grown past the single `_nemo.py` driver into a multi-stage, **file-handoff** pipeline plus separate retrieval-evaluation scripts. Stages communicate only through JSON files on disk; there is no in-memory orchestration across scripts.

## Pipeline Overview

There are **two SDG routes**, selected by which `_nemo.py` flag you run:

**Route A — in-process SDG (`--sdg` → `--prep`), all inside `_nemo.py`:**
```
_nemo.py --sdg   → per-doc -sdg.json + {output_dir}/full_sdg_output.json
_nemo.py --prep  → reads full_sdg_output.json → embed_data_prep/ splits + eval_beir/
```

**Route B — logical-chunk SDG (`--sdg-logical`), hands off to standalone scripts:**
```
1. _nemo.py --sdg-logical --cfg cfg/nemo.toml
     → doc-chunks_{size}_random_logical/{doc}-chunks.json, -logic-chunks.json,
       -relevance.json (if relevance_filter), -logic-ctx.json
2. extract_artifacts.py --cfg extract_artifacts.toml
     → {doc}-logic-artifacts.json   (span-level langextract + chunk-level ChunkSignals)
3. generate-qa.py --config generate-qa.toml
     → qa-gen/generated-questions.json (+ _qa_only.json, .csv)
4. self-check/self-check-qa.py --config ./self-check-qa.toml   (run from self-check/)
     → self-check-output/self-check-qa-results.json (+ _wo_context.json, .csv)
```
Route B's downstream scripts are **single-file standalone — they do NOT import `aisa/`**. `--sdg-logical` does **not** write `full_sdg_output.json`, so `--prep` does not follow it; Route B's "data prep" is the `extract_artifacts.py → generate-qa.py → self-check-qa.py` chain instead.

**Retrieval evaluation (off the main path, optional):**
```
filter-questions-citation-eval.py  → drops QA records with empty/sentinel citations
eval.py                            → pub242-vs-techbriefs Qdrant retrieval eval
eval2.py                           → base-vs-fine-tuned embedding eval (techbriefs)
```

## Entry Points & Commands

Install: `pip install -r reqs.txt` (now includes `langextract`, `faiss-cpu`, `datasets`, `qdrant`-via-`sentence_transformers`, `tqdm`). No test suite, lint config, or build script exists.

### `_nemo.py` (chunking + Route A/B SDG; config: `cfg/nemo.toml`, TOML)

```bash
# Chunking only — path2chunks on every .md under input_dir, then stop
python _nemo.py --chunk-only --cfg cfg/nemo.toml --input_dir <md_dir> --output_dir <out_dir>

# Route A: in-process SDG (QA pairs + LLM-as-judge eval) → full_sdg_output.json
python _nemo.py --sdg --cfg cfg/nemo.toml

# Route A: data prep (filter, hard-negative mining, unroll, split)
python _nemo.py --prep --cfg cfg/nemo.toml

# Route B: logical chunking + 1:1 logical contexts (random_logical only)
python _nemo.py --sdg-logical --cfg cfg/nemo.toml

# Multiple stages sequentially
python _nemo.py --sdg --prep --cfg cfg/nemo.toml
```

At least one of `--chunk-only` / `--sdg` / `--sdg-logical` / `--prep` must be passed (`parser.error` otherwise). The `__main__` block builds `cfg["nemo_task"] = {"chunk", "sdg", "sdg_logical", "prep"}` from these flag booleans; `--cfg` defaults to `./cfg/nemo.toml`. `--input_dir`/`--output_dir` override `[general].data_dir`/`[general].output_dir`. The task name `"chunk"` maps to `run_chunk_only_pipeline`, `"sdg"`→`run_sgd_pipeline`, `"sdg_logical"`→`run_sgd_logical_pipeline`, `"prep"`→`run_data_prep_pipeline` in `QAGenerator.tasks`.

`--sdg-logical` **requires `[chunking].method == "random_logical"`** (raises `ValueError` otherwise; for `logical` mode use `--sdg`).

### `extract_artifacts.py` (Route B step 2; config: `extract_artifacts.toml`)

```bash
python extract_artifacts.py --cfg extract_artifacts.toml [--input_dir <dir>] [--overwrite]
```
Reads `*-logic-ctx.json`, writes `*-logic-artifacts.json` (span-level via `langextract`→gpt-4o-mini with a 21-class normative taxonomy + verbatim spans; chunk-level via OpenAI Structured Outputs→`ChunkSignals` Pydantic object). Default `--cfg` is `./extract_artifacts.toml` (shape: `[paths]` + `[artifact_extraction]`). It **also** accepts the project-wide `cfg/nemo.toml`: in that case it reads `[general].output_dir`, auto-discovers the single `doc-chunks_*_random_logical/` dir, and uses the `[langextract]` section. Standalone — no `aisa/` import.

### `generate-qa.py` (Route B step 3; config: `generate-qa.toml`)

```bash
python generate-qa.py --config generate-qa.toml [--host localhost] [--port 11434]
```
Two phases: (1) generate 3–5 questions per (context, artifact); (2) extract a verbatim citation per question. Config section `[generate-qa]`. Supports `gpt-*` (OpenAI structured outputs), `gemini-*` (Google), and any other model name (Ollama; `--host/--port`). Standalone.

### `self-check/self-check-qa.py` (Route B step 4; config: `self-check/self-check-qa.toml`)

```bash
cd self-check && python self-check-qa.py --config ./self-check-qa.toml
```
**Ollama-only** LLM-as-judge over `generate-qa.py` output; per-record `evaluation ∈ {0, 0.5, 1}` (`-1` = no context, skipped). Config section `[self-check-qa]`; **all paths are relative to the `self-check/` directory** — run the script from there. Standalone. `cluster/self-check-qa.slurm` runs it on a cluster.

### Retrieval-eval scripts

```bash
python filter-questions-citation-eval.py --cfg filter-questions-citation-eval.toml
python eval.py  --cfg eval.toml  [--rebuild] [--mode all|pub242_only|techbriefs_only|both] [-n N] [--selection sequential|random] [--seed S] [--top-k K]
python eval2.py --config eval2.toml [--setup-rag] [--batch-rag] [--setup-rag-ft] [--batch-rag-ft] [--top-k K] [-n N]
```
- `filter-questions-citation-eval.py`: drops QA records whose `full_citation.citation` is empty or matches a code/prompt "no citation" sentinel; emits `<stem>-c-eval.json` + `<stem>-c-eval-dropped.json`. Its output feeds `eval.toml`/`eval2.toml`.
- `eval.py`: embeds pub242 + techbriefs answers into one Qdrant collection (distinguished by a `source` payload field) and queries with pub242 questions under three pool-restriction modes. Outputs `eval_{mode}.json/.csv`.
- `eval2.py`: base vs fine-tuned embedding eval on the techbriefs corpus (Qdrant); paths are **CWD-relative** (run from project root). Outputs `{name}_{rag|rag_ft}_results.json/.csv`. The `--setup-*` flags build the Qdrant collections; `--batch-*` run the queries.

## Runtime Requirements

- **API keys** via `.env`: `OPENAI_API_KEY`, `GOOGLE_API_KEY`. `aisa/gen/providers.py` loads `.env`; missing keys for a selected provider raise at `BaseInfo.__init__`. `extract_artifacts.py`/`generate-qa.py` independently `load_dotenv()` and need `OPENAI_API_KEY` when using gpt-* models.
- **Ollama**: if the chosen chat/embed model isn't in `CHAT_MODELS`/`EMBED_MODELS` in `providers.py`, the code falls back to Ollama (`http://localhost:11434`) or HuggingFace. `check_existing_model` in `ollama_api.py` `exit()`s if the model isn't installed locally. `_nemo.py` import constructs an `Embedder` default → `ollama_api.list_models()`, so **Ollama must be reachable for any `_nemo.py` run**, even `--sdg-logical` which never embeds. `generate-qa.py` and `self-check-qa.py` use the `ollama` client directly for non-gpt/gemini models.
- **Prompts** live in `prompt_lib` (`./prompts`). `BaseLLM.read_prompt(name)` reads `{prompt_lib}/{name}.txt`. Required prompts depend on the route:
  - `--sdg` (Route A): `nemo_artifacts`, `nemo_qa-gen`, `nemo_eval`.
  - `[chunking].method = "logical"`: `nemo_logical-chunk-02` (used by `LLMSemanticChunker`).
  - `[chunking].method = "random_logical"`: `nemo_logical-chunk` (used by `HybridLogicalChunker`).
  - `[chunking].relevance_filter = true` (random_logical only): `nemo_eval-02` (used by `evaluate_chunks`).
  - `extract_artifacts.py`: span + chunk prompts, default `nemo_logic-artifacts-04-span` / `nemo_logic-artifacts-04-chunk` (configurable via `prompt_name` / `chunk_prompt_name`).
  - `generate-qa.py`: `nemo_qa-gen-artifact` (questions) + `nemo_extract-citation` (citations).
  - `self-check/self-check-qa.py`: `self-check/prompts/self-check-01.txt`.

  Missing required prompts raise `FileNotFoundError`. The `prompts/` dir holds several versioned variants (`-02`, `-03`, `-04`, `-span`/`-chunk`) — check the active config before assuming a name.

## Architecture

### 1. `aisa/` — reusable LLM/embedding/parsing toolkit (used by `_nemo.py` only)

- **`aisa/gen/`** — provider-agnostic chat + embedding wrappers
  - `providers.py` — `Provider` enum (OpenAI/Google/Ollama/HuggingFace), per-model price/token bookkeeping (`ChatInfo`, `EmbedInfo`), registries `CHAT_MODELS` / `EMBED_MODELS`. **Adding a priced model = adding an entry here.**
  - `chat_llm.py` — `BaseLLM` wraps LangChain `ChatOpenAI` / `ChatGoogleGenerativeAI` / `ChatOllama` (`_init_ollama_model` uses `ChatOllama(format="json")` when `json_mode`). `run_chain` uses `abatch` for async batching; `json_mode=True` routes responses through `clean_json` which strips ```json fences. Hosts `read_prompt`.
  - `embed.py` — `Embedder` mirrors the pattern for embeddings; falls back to HuggingFace for unknown names. `HFEmbedder` is `# NOTE: DEPRECATED`.
  - `ollama_api.py` — `list_models` / `check_existing_model` (the latter `exit()`s on a missing local model).
  - `prompts.py` — `get_token_count` and prompt helpers; imported by `chunkers.py`.
  - `decorators.py` — `ChatResponse` / `EmbedResponse` decorators auto-track tokens, cost, timing by mutating `self.info`. **Do not bypass them if you care about the cost log.**
- **`aisa/parse/`** — document chunking
  - `chunk.py` — (a) `Chunker` aggregates a `ParsedDoc` into section/subsection `Chunk` records (fuzzy `fuzz_position`, rapidfuzz `partial_ratio`); (b) `RecursiveChunker` is a token-budgeted **batcher** for LLM chain inputs (packs small chunks under `max_input_tokens`), not a text splitter. A commented-out older `fuzz_position` precedes the live one.
  - `chunkers.py` — pluggable *text splitters* + the logical-grouping engine. `RecursiveTextChunker` (LangChain `RecursiveCharacterTextSplitter`), `LLMSemanticChunker` (mode `logical`; 50-token pre-split + LLM windowing via `nemo_logical-chunk-02`, `logical_window`/`logical_stride`), `HybridLogicalChunker` (mode `random_logical`; recursive pre-split → LLM grouping via `nemo_logical-chunk`, `hybrid_window`/`hybrid_stride`, exposes `last_recursive_pieces`/`last_source_indices`). Module-level `group_kept_pieces(...)` does **mask-aware** logical grouping over a filtered subset of recursive pieces (used by the relevance filter — see below); gaps between kept runs are implicit hard splits. `get_chunker(chunk_cfg, llm=None)` is the factory. **Don't confuse with `chunk.py::RecursiveChunker` — that one batches for LLM calls; these split text.**
  - `doc.py` / `naming.py` — parse structured filenames + external CSV metadata (`_main.csv`, `*-metadata.csv`) to attach section/subsection titles.
- **`aisa/utils/`** — `files.py` (json/csv/toml I/O, path normalization), `log.py` (loguru with custom levels `NLP`/`CHUNK`/`TIME`/`MODEL`/`COST`/`RESP`), `types.py` (`dictlist` = `list[dict]`), `helpers.py`. Imported as `from aisa.utils import files, logger, dictlist`.

### 2. `_nemo.py` — `QAGenerator`

Holds **two distinct downstreams** plus a relevance filter.

**`path2chunks`** — splits markdown into token-sized chunks (`tiktoken` `gpt-3.5-turbo` encoding), stripping tables/images via `MD_PATTERNS` first. Chunker chosen by `[chunking].method` (`recursive` / `logical` / `random_logical`), built via `aisa.parse.chunkers.get_chunker`. Output dir `{output_dir}/doc-chunks_{size}_{method}/` (method in the path so switching methods doesn't reuse stale chunks). `random_logical` additionally writes `{doc}-logic-chunks.json` (final logical chunks with `source_chunk_ids` provenance) and uses **that** file as the cache key; `{doc}-chunks.json` holds the recursive intermediate.

**Relevance filter (`random_logical` only)** — when `[chunking].relevance_filter = true`, `QAGenerator.__init__` eagerly constructs an `AsyncOpenAI` client (**raises `RuntimeError` without `OPENAI_API_KEY`**). `path2chunks` then: recursive pre-split → `evaluate_chunks` scores every piece `0/0.5/1` for pavement relevance (one `gpt-4o-mini` call per piece via `nemo_eval-02`, concurrency-bounded by `relevance_concurrency`, cached to `{doc}-relevance.json`) → `group_kept_pieces` groups only pieces scoring `> 0.5`. On eval failure it logs and keeps all (`score=1.0`). The filter is silently ignored for non-`random_logical` methods.

**Route A — `run_sgd_pipeline` (`--sdg`):** per `.md`: `path2chunks` → `extract_artifacts` (packs chunks into `RecursiveChunker` bundles, de-overlaps neighbors with `_trim_overlap_for_context`, calls `nemo_artifacts`; writes `-ctx.json` + `-artifacts.json`) → `generate_qa_pairs` (`facts_block` + `context_block`, `nemo_qa-gen`, hardcoded multi-hop/structural/contextual × factual/relational/inferential/temporal/procedural counts; `-qa_pairs.json`) → `evaluate_qa_pairs` (LLM-as-judge `nemo_eval`; `-qa_eval.json`). Per-doc `-sdg.json`; corpus-wide `full_sdg_output.json`. **`QAGenerator.extract_artifacts` (in-process) is a different thing from the standalone `extract_artifacts.py`** — see Gotchas.

**Route A — `run_data_prep_pipeline` (`--prep`):** consumes `full_sdg_output.json`:
- `filter_and_convert` — drops QA pairs with `overall.score < quality_threshold` (default 7.0); `pos_doc` = chunks whose `chunk_id` ∈ `segment_ids`.
- `mine_hard_negatives` — embeds unique positives + queries, top-k cosine, excludes true positives and anything above `hard_neg_margin` (default 0.95, to skip near-duplicates).
- `unroll_pos_docs` — splits multi-positive rows into 1:1 rows with suffixed `question_id`s.
- `save_splits` — 80/10/10 random split into `{output_dir}/embed_data_prep/`; also emits BEIR `eval_beir/{corpus,queries}.jsonl` + `test.tsv`.

**Route B — `run_sgd_logical_pipeline` (`--sdg-logical`):** requires `method == "random_logical"`. Per `.md`: `path2chunks` → `_build_logical_contexts` writes `{doc}-logic-ctx.json` (1:1 logical-chunk → context, with `u_ctx_id`/`source_u_logic_chunk_ids`, token-budget warnings vs `max_input_tokens`). No LLM QA/eval here — it stops at logical contexts and hands off to the standalone scripts.

### 3. Standalone Route-B / eval scripts (no `aisa/` import)

`extract_artifacts.py`, `generate-qa.py`, `self-check/self-check-qa.py`, `eval.py`, `eval2.py`, `filter-questions-citation-eval.py` are each single-file: they re-declare their own IO helpers, loguru levels, and provider routing rather than importing the toolkit. Change them in place; do not assume `aisa/` edits propagate to them. `cluster/` and `self-check/cluster/` hold `.slurm` jobs to run stages on a cluster; `models.txt`/`pull_models.sh` pull Ollama models there.

## Configuration

There is **one TOML per script**, not a single global config:

| Script | Config | Sections it reads |
|---|---|---|
| `_nemo.py` | `cfg/nemo.toml` | `[general]`, `[llm]`, `[embedding]`, `[chunking]` |
| `extract_artifacts.py` | `extract_artifacts.toml` (or `cfg/nemo.toml`) | `[paths]` + `[artifact_extraction]` — **or** `[general].output_dir` + `[langextract]` if given `cfg/nemo.toml` |
| `generate-qa.py` | `generate-qa.toml` | `[generate-qa]` |
| `self-check/self-check-qa.py` | `self-check/self-check-qa.toml` | `[self-check-qa]` (paths relative to `self-check/`) |
| `eval.py` | `eval.toml` | `[corpus]`,`[queries]`,`[embedding]`,`[qdrant]`,`[eval]` |
| `eval2.py` | `eval2.toml` | `[general]`,`[rag]`,`[rag-ft]` (CWD-relative paths) |
| `filter-questions-citation-eval.py` | `filter-questions-citation-eval.toml` | `[filter-questions-citation-eval]` |

`cfg/nemo.toml` mixes worlds: `[general]`/`[llm]`/`[embedding]`/`[chunking]` are read by `_nemo.py`; `[langextract]` is read **only by `extract_artifacts.py`** when it is handed `cfg/nemo.toml` (not by `_nemo.py`). `[pub242]`, `[jsa]`, `[nlp]`, `[doc]`, `[general.tasks]`, `[general.theme]`, and `[general].metadata_folder` are carryovers from a larger sibling project and are **not read by any script here**. If `[chunking]` is absent, `QAGenerator` falls back to `method = "recursive"` with `chunk_size`/`recursive_overlap` from `[embedding]`. `[chunking]` keys: `method`, `chunk_size`, `recursive_overlap`, `logical_*` (logical), `hybrid_window`/`hybrid_stride` (random_logical), `relevance_filter`/`relevance_concurrency` (random_logical).

## Key behavioral conventions

- **Idempotent file-cached stages**: nearly every method checks `Path(out).exists() and not self.overwrite` before re-running. In `_nemo.py` there is **no config flag** for this — set `self.overwrite = True` in `QAGenerator.__init__` or delete the output. `extract_artifacts.py` has an `--overwrite` flag. The `random_logical` cache key is `{doc}-logic-chunks.json` (not `-chunks.json`).
- **Hardcoded generation counts**: `query_counts_*`, `reasoning_counts_*`, `min_hops`, `max_hops`, `min_complexity`, `num_pairs`, `max_artifacts` are set in `QAGenerator.__init__`, not in TOML — edit the class.

## Deprecated / Reference / Forward-looking

- `_depr/nemotron/` — earlier standalone-script version (`sdg.py`, `data_prep.py`, README). Reference only; `_nemo.py` is current. Don't edit unless asked.
- `aisa/gen/embed.py::HFEmbedder` is `# NOTE: DEPRECATED`.
- `plans/*.md`, `cfg/nemo_specs.toml`, `rawdata-pubs/` (all git-untracked) describe a **proposed, not-yet-implemented** refactor to run the whole Route-B pipeline on local Ollama (`gpt-oss:20b`). The SRS status is "Proposed" — e.g. `_nemo.py::evaluate_chunks` still hardcodes `model="gpt-4o-mini"` and `__init__` still requires `OPENAI_API_KEY`. Treat those docs as design intent, not as a description of current code.

## Gotchas

- **Two `extract_artifacts`**: `QAGenerator.extract_artifacts` (in-process, Route A `--sdg`, `nemo_artifacts` prompt, `-artifacts.json`) vs the standalone `extract_artifacts.py` (Route B step 2, langextract + `ChunkSignals`, `-logic-artifacts.json`). Same name, unrelated code paths.
- **`--sdg` and `--sdg-logical` are different downstreams.** `--prep` only follows `--sdg` (it reads `full_sdg_output.json`, which `--sdg-logical` never writes). After `--sdg-logical`, continue with `extract_artifacts.py → generate-qa.py → self-check-qa.py`.
- **Relevance filter eager init**: enabling `[chunking].relevance_filter` constructs `AsyncOpenAI` at `QAGenerator.__init__`, so even a `--chunk-only` run will fail fast without `OPENAI_API_KEY`.
- `BaseLLM`/`Embedder` instantiate their provider client at construction. Changing `cfg.model` on an existing instance does not re-init the underlying LangChain model.
- `run_chain` passes the raw `prompt` as a `PromptTemplate` with `input_variables=list(input_docs[0].keys())` — every dict in the batch must have the same keys, matching the `{placeholders}` in the prompt file. Missing keys fail silently as literal `{foo}`.
- `self-check/self-check-qa.py` resolves all config paths relative to `self-check/` — run it from that directory or paths break.
- `data/`, `data.0`, `data.1`, `small_corpus`, `logs`, `.DS_Store`, `rawdata-pub*`, and `self-check/self-check-output-cluster` are gitignored; the `rawdata*` paths referenced in the configs are not in this repo.
