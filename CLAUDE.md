# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Custom implementation of NVIDIA's Nemotron Embed Recipe for synthetic data generation (SDG) and data preparation, targeting fine-tuning of embedding models on domain-specific `.md` corpora. Unlike the reference recipe it does not depend on NeMo distributed runtime — it plugs into the project's own `aisa` LLM/embedder abstractions.

## Entry Point & Commands

The pipeline is driven by `_nemo.py` at repo root. Config is TOML-based (`cfg/nemo.toml`).

```bash
# Chunking only — runs path2chunks on every .md under input_dir, writes -chunks.json, stops
python _nemo.py --chunk-only --cfg cfg/nemo.toml --input_dir <md_dir> --output_dir <out_dir>

# Stage 0 — Synthetic Data Generation (QA pairs + LLM-as-judge eval)
python _nemo.py --sdg --cfg cfg/nemo.toml --input_dir <md_dir> --output_dir <out_dir>

# Stage 1 — Data Prep (filter, hard-negative mining, unroll, split)
python _nemo.py --prep --cfg cfg/nemo.toml --output_dir <out_dir>

# Multiple stages sequentially
python _nemo.py --sdg --prep --cfg cfg/nemo.toml
```

At least one of `--chunk-only` / `--sdg` / `--prep` must be passed. The `__main__` block builds `cfg["nemo_task"]` from these flags; `--cfg` defaults to `./cfg/nemo.toml`.

`--input_dir`/`--output_dir` override `[general].data_dir` / `[general].output_dir` in the TOML. Stage 1 reads `{output_dir}/full_sdg_output.json` produced by Stage 0 — the two stages communicate via that file, not in-memory.

Install: `pip install -r reqs.txt`. No test suite, lint config, or build script exists.

## Runtime Requirements

- **API keys** via `.env` (loaded by `aisa/gen/providers.py` with `python-dotenv`): `OPENAI_API_KEY`, `GOOGLE_API_KEY`. Missing keys for a selected provider raise at `BaseInfo.__init__`.
- **Ollama**: if the chosen chat/embed model isn't in the hardcoded `CHAT_MODELS`/`EMBED_MODELS` registries in `providers.py`, the code falls back to Ollama (`http://localhost:11434`) or HuggingFace. `check_existing_model` in `ollama_api.py` `exit()`s if the model isn't installed locally.
- **Prompts**: `BaseLLM.read_prompt(name)` reads `{llm.prompt_lib}/{name}.txt`. `_nemo.py` expects prompts named `nemo_artifacts`, `nemo_qa-gen`, `nemo_eval` — these must exist in the `prompt_lib` dir or Stage 0 raises `FileNotFoundError`. `nemo_logical-chunk` is additionally required when `[chunking].method = "logical"`. The prompt library points to `./prompts`.

## Architecture

Two conceptual layers:

### 1. `aisa/` — reusable LLM/embedding/parsing toolkit

- **`aisa/gen/`** — provider-agnostic chat + embedding wrappers
  - `providers.py` defines the `Provider` enum (OpenAI/Google/Ollama/HuggingFace), per-model price/token bookkeeping (`ChatInfo`, `EmbedInfo`), and registry dicts `CHAT_MODELS` / `EMBED_MODELS`. **Adding a priced model = adding an entry here.**
  - `chat_llm.py` — `BaseLLM` wraps LangChain's `ChatOpenAI` / `ChatGoogleGenerativeAI` / `ChatOllama`. `run_chain` uses `abatch` for async batching. `json_mode=True` routes responses through `clean_json` which strips ```json fences.
  - `embed.py` — `Embedder` mirrors the same pattern for embeddings; falls back to HuggingFace for unknown model names.
  - `decorators.py` — `ChatResponse` / `EmbedResponse` decorators automatically track token counts, cost, and timing on every call by mutating `self.info`. **Do not bypass the decorators if you care about the cost log.**
- **`aisa/parse/`** — document chunking
  - `chunk.py` has two distinct pieces: (a) `Chunker` — aggregates a `ParsedDoc` into section/subsection `Chunk` records using fuzzy-matched subsection boundaries (`fuzz_position` with rapidfuzz `partial_ratio`); (b) `RecursiveChunker` — token-budgeted **batcher** for LLM chain inputs, not a text splitter despite the name. `_nemo.py` uses the latter to pack multiple small chunks into each LLM call under `max_input_tokens`.
  - `chunkers.py` — pluggable *text splitters* used by `_nemo.py::path2chunks`. Exposes `RecursiveTextChunker` (LangChain `RecursiveCharacterTextSplitter` wrapper) and `LLMSemanticChunker` (LLM-driven logical chunking via the `nemo_logical-chunk` prompt, windowed with `logical_window`/`logical_stride`). `get_chunker(chunk_cfg, llm)` is the factory. **Don't confuse with `chunk.py::RecursiveChunker` — that one batches for LLM calls, these ones split text.**
  - `doc.py` / `naming.py` — parses structured filenames + external CSV metadata (`_main.csv`, `*-metadata.csv`) to attach section/subsection titles to documents.
- **`aisa/utils/`** — `files` (json/csv/toml I/O, path normalization), `logger` (loguru-based with custom levels like `"NLP"`, `"CHUNK"`, `"TIME"`, `"MODEL"`, `"COST"`, `"RESP"`), `dictlist` alias (`list[dict]`).

### 2. `_nemo.py` — the SDG + data-prep pipeline

`QAGenerator` orchestrates four LLM stages per `.md` file:

1. **`path2chunks`** — splits markdown into token-sized chunks (`tiktoken` `gpt-3.5-turbo` encoding), stripping tables/images via `MD_PATTERNS` first. Chunker implementation is selected via `[chunking].method` (`"recursive"` or `"logical"`), built in `QAGenerator.__init__` via `aisa.parse.chunkers.get_chunker`. Output: `{output_dir}/doc-chunks_{size}_{method}/{doc_id}-chunks.json` (method is part of the cache path so switching methods doesn't reuse stale chunks).
2. **`extract_artifacts`** — packs chunks into `RecursiveChunker` bundles, de-overlaps neighbors with `_trim_overlap_for_context` (removes text that would otherwise be double-counted after `RecursiveCharacterTextSplitter`'s overlap), calls the `nemo_artifacts` prompt. Writes `-ctx.json` (de-overlapped bundles with token counts) and `-artifacts.json`.
3. **`generate_qa_pairs`** — builds per-bundle `facts_block` (from artifacts) + `context_block` (from `-ctx.json`) and runs `nemo_qa-gen` with hardcoded counts for multi-hop/structural/contextual queries × factual/relational/inferential/temporal/procedural reasoning types. Output: `-qa_pairs.json`.
4. **`evaluate_qa_pairs`** — LLM-as-judge via `nemo_eval`. Output: `-qa_eval.json`. The combined per-doc result goes to `-sdg.json`; across-corpus aggregation to `full_sdg_output.json`.

`run_data_prep_pipeline` (Stage 1) consumes `full_sdg_output.json` and:
- **`filter_and_convert`** — drops QA pairs with `overall.score < quality_threshold` (default 7.0); assembles training records with `pos_doc` = chunks whose `chunk_id` appears in the QA pair's `segment_ids`.
- **`mine_hard_negatives`** — embeds all unique positive passages + all queries, takes top-k by cosine, excludes true positives, excludes anything above `hard_neg_margin` (default 0.95 — to avoid mining near-duplicates of positives).
- **`unroll_pos_docs`** — splits multi-positive rows into one-positive rows with suffixed `question_id`s so downstream biencoder training sees 1:1 mappings.
- **`save_splits`** — 80/10/10 random split; also emits BEIR-format `eval_beir/{corpus,queries}.jsonl` + `test.tsv`.

### Key behavioral conventions

- **Idempotent file-cached stages**: nearly every method checks `Path(base_out).exists() and not self.overwrite` before re-running. To force regeneration, set `self.overwrite = True` in `QAGenerator.__init__` or delete the output file. There is no config flag for this.
- **Hardcoded generation counts**: `query_counts_*`, `reasoning_counts_*`, `min_hops`, `max_hops`, `num_pairs` are set in `QAGenerator.__init__`, not in TOML — edit the class to change.
- **`cfg/nemo.toml` mixes two worlds**: the `[general]`, `[llm]`, `[embedding]`, `[chunking]` sections are used by `_nemo.py`. The `[pub242]`, `[jsa]`, `[nlp]`, `[doc]` sections and `[general.tasks]` flags are carryovers from a larger sibling project and are **not read by `_nemo.py`**. If `[chunking]` is absent, `QAGenerator` falls back to `method = "recursive"` with `chunk_size`/`recursive_overlap` taken from `[embedding]` (backward compat).

## Deprecated / Reference Code

- `_depr/nemotron/` — earlier standalone-script version of the same pipeline (`sdg.py`, `data_prep.py`) plus its README. Kept for reference; `_nemo.py` is the current implementation. Don't edit `_depr/` unless explicitly asked.
- `aisa/gen/embed.py::HFEmbedder` is marked `# NOTE: DEPRECATED`.
- `aisa/parse/chunk.py` has a commented-out older `fuzz_position` — the live version is the one below it.

## Gotchas

- `BaseLLM`/`Embedder` instantiate their provider client at construction time. Changing `cfg.model` on an existing instance does not re-init the underlying LangChain model.
- `run_chain` passes the raw `prompt` string as a `PromptTemplate` with `input_variables=list(input_docs[0].keys())` — every dict in the batch must have the same keys, and those keys must match the `{placeholders}` in the prompt file. Missing keys fail silently as literal `{foo}` in the prompt.
- The `__main__` block parses args once into `args`, overrides `[general].output_dir`/`data_dir` from the CLI if provided, and synthesizes `cfg["nemo_task"] = {"chunk": ..., "sdg": ..., "prep": ...}` from the flag booleans. The task name `"chunk"` maps to `run_chunk_only_pipeline` in `QAGenerator.tasks`.
- `data/` and `.DS_Store` are gitignored; the `rawdata*` paths referenced in `cfg/nemo.toml` are not in this repo.
