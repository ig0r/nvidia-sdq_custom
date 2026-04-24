# Software Requirements Specification: Logical Chunking

**Feature:** Configurable LLM-based logical chunking for the Nemotron SDG pipeline
**Component:** `nvidia-sdq_custom`
**Version:** 0.1 (draft)
**Status:** Proposed

---

## 1. Introduction

### 1.1 Purpose
This SRS defines requirements for adding a second, LLM-driven chunking strategy (`LLMSemanticChunker`) to the SDG pipeline, selectable at runtime via configuration. It extends, but does not replace, the existing `RecursiveCharacterTextSplitter`-based chunking used in `_nemo.py::QAGenerator.path2chunks`.

### 1.2 Scope
In scope:
- New chunker implementations and a selection factory in `aisa/parse/chunkers.py`.
- New `[chunking]` TOML section in `cfg/nemo.toml`.
- Integration with `QAGenerator.path2chunks` only. Downstream stages (artifact extraction, QA generation, evaluation, data prep) are unaffected by design.
- A new prompt file `prompts/nemo_logical-chunk.txt`.

Out of scope:
- Changes to `aisa/parse/chunk.py::Chunker` (the `ParsedDoc` section aggregator) or its `RecursiveChunker` batcher.
- Changes to `Embedder`, `EmbedConfig`, or the embedding pipeline.
- Changes to deprecated code under `_depr/`.

### 1.3 Definitions
- **Chunker** — a component producing a list of text chunks from a single markdown document.
- **Recursive chunker** — token-budgeted character splitter (LangChain `RecursiveCharacterTextSplitter`).
- **Logical chunker** — LLM-driven splitter that places boundaries at semantic shifts.
- **Pre-split piece** — small fixed-size fragment produced before LLM tagging.
- **Window** — contiguous range of pre-split pieces sent to the LLM in a single call.
- **Stride** — number of pre-split pieces between consecutive window starts.

### 1.4 References
- `plans/plan-logical-chunking.md` — companion implementation plan.
- `CLAUDE.md` — project conventions and architecture.
- Reference implementation: `chunking_evaluation/chunking/llm_semantic_chunker.py` (Brandon Starxel, GitHub).

---

## 2. Overall Description

### 2.1 Product Perspective
The SDG pipeline (`_nemo.py`) runs four LLM-backed stages per markdown file. Stage 0.1 (`path2chunks`) currently does character-based splitting with a fixed chunk size / overlap. This feature inserts a selectable chunker at that stage. All later stages consume the same `{text, chunk_id, tokens}` chunk schema and require no modification.

### 2.2 User Classes
- **Pipeline operator** — runs `_nemo.py --sdg`. Chooses chunking method via `[chunking].method`.
- **Pipeline developer** — extends or tunes chunker behavior; adds prompts.

### 2.3 Operating Environment
- Python 3.x; dependencies already declared in `reqs.txt` (`langchain`, `tiktoken`, `loguru`, provider SDKs).
- Requires a working `BaseLLM` provider and a populated `prompt_lib` when `method = "logical"`.

### 2.4 Constraints
- Token counting MUST use the existing `get_token_count` helper (tiktoken, `gpt-3.5-turbo` encoding) to stay consistent with all downstream token math.
- Chunker output schema MUST match `{"text": str, "chunk_id": int, "tokens": int}` exactly — this is what `extract_artifacts`, `-ctx.json`, QA generation, and `filter_and_convert` assume.
- The existing file-cache idiom (`if Path(out).exists() and not self.overwrite`) MUST be preserved.
- No changes may bypass the `ChatResponse` decorator's cost tracking.
- LLM response handling MUST follow the existing repo pattern: provider-native `json_mode=True` + `aisa.gen.prompts.clean_json` for parsing + manual field validation with `.get(...)` defaults. Pydantic models MUST NOT be introduced for LLM response validation; pydantic remains reserved for config (`LLMConfig`, `EmbedConfig`) and internal data schemas (`Chunk`, `StdName`).

### 2.5 Assumptions
- The user maintains the `prompt_lib` directory and will add `nemo_logical-chunk.txt` before enabling logical mode.
- The selected LLM supports `json_mode` (all currently registered models in `CHAT_MODELS` do).
- Documents are markdown; table/image stripping (`MD_PATTERNS`) happens before chunking, as it does today.

---

## 3. Functional Requirements

### FR-1 Configurable chunking method
**FR-1.1** The system SHALL accept a new TOML section `[chunking]` in `cfg/nemo.toml`.
**FR-1.2** `[chunking].method` SHALL accept the values `"recursive"` and `"logical"`.
**FR-1.3** `[chunking].method = "recursive"` SHALL produce chunks byte-for-byte identical to the current `path2chunks` output given the same inputs, config, and tiktoken version.
**FR-1.4** An unknown `method` value SHALL raise a clear `ValueError` naming the offending value and the allowed set.
**FR-1.5** A missing `[chunking]` section SHALL be treated as `method = "recursive"` with defaults inherited from the current `[embedding].chunk_size` / `[embedding].recursive_overlap` to preserve existing behavior for unmodified configs.

### FR-2 Chunker interface
**FR-2.1** The system SHALL expose a chunker abstraction in `aisa/parse/chunkers.py` with a `split(text: str) -> list[str]` method.
**FR-2.2** The system SHALL provide two concrete implementations: `RecursiveTextChunker` and `LLMSemanticChunker`.
**FR-2.3** The system SHALL provide a factory `get_chunker(chunk_cfg: dict, llm: BaseLLM | None = None) -> Chunker` that returns the correct implementation and raises when `method = "logical"` but `llm is None`.
**FR-2.4** The existing `aisa/parse/chunk.py::RecursiveChunker` (the token-budgeted LLM batcher) SHALL remain untouched and retain its name and call site at `_nemo.py:194`.

### FR-3 Recursive chunker (parity)
**FR-3.1** `RecursiveTextChunker` SHALL wrap `langchain.text_splitter.RecursiveCharacterTextSplitter` with `chunk_size`, `chunk_overlap`, and `length_function=get_token_count` sourced from `[chunking]`.
**FR-3.2** `RecursiveTextChunker.split` SHALL return `list[str]`; numeric `chunk_id` and `tokens` fields are added by `path2chunks`, not by the chunker, to keep the chunker interface narrow.

### FR-4 Logical chunker — pre-splitting
**FR-4.1** `LLMSemanticChunker` SHALL pre-split input text using `RecursiveCharacterTextSplitter(chunk_size=logical_presplit_tokens, chunk_overlap=0, length_function=get_token_count)`.
**FR-4.2** Each pre-split piece SHALL be wrapped as `<start_chunk_{i}>{text}<end_chunk_{i}>` where `i` is its zero-based index in the document.

### FR-5 Logical chunker — windowed LLM calls
**FR-5.1** The system SHALL issue one LLM call per window of `logical_window` consecutive tagged pieces.
**FR-5.2** Consecutive windows SHALL advance by `logical_stride` pieces (`logical_stride <= logical_window` required; violation raises at construction time).
**FR-5.3** Each call SHALL use the `nemo_logical-chunk` prompt loaded via `BaseLLM.read_prompt`.
**FR-5.4** Each call SHALL produce a JSON response of the form `{"split_after": [int, ...]}` consumed via `BaseLLM.query` (or `run_chain` for async batch; implementation detail).
**FR-5.5** LLM cost and token accounting SHALL flow through the existing `ChatResponse` decorators without modification.
**FR-5.6** The LLM MUST be invoked with `LLMConfig.json_mode = True` (already the project default). Response parsing SHALL go through `aisa.gen.prompts.clean_json`, which is already applied inside `BaseLLM.query` / `run_chain`; no additional JSON parsing layer is permitted.

### FR-6 Logical chunker — response handling
**FR-6.1** Response validation SHALL use manual `.get(...)` / `isinstance(...)` checks on the dict returned by `clean_json`. Pydantic (or any other schema-validation library) SHALL NOT be used to validate the LLM response, to stay consistent with the rest of the pipeline (`get_fact_blocks`, `filter_and_convert`, etc., which all read LLM output defensively via `.get`).
**FR-6.2** The system SHALL treat the response as invalid if any of the following hold: `clean_json` returned `{}` (empty dict — its documented failure sentinel), the result is not a `dict`, the `split_after` key is missing, or `split_after` is not a `list`.
**FR-6.3** The system SHALL parse `split_after` by keeping only elements that are `int` (via `isinstance`), within `[window_start, window_end)`, and strictly greater than the previous kept index. Non-conforming elements are silently dropped.
**FR-6.4** On complete parse failure for a window (FR-6.2 triggered, or no valid indices remained after FR-6.3), the system SHALL log a `CHUNK`-level warning and force a single split at the window end.
**FR-6.5** Split points near a window boundary SHALL be de-duplicated across overlapping windows so that consecutive calls do not produce two splits at the same absolute piece index.

### FR-7 Logical chunker — assembly
**FR-7.1** Final chunks SHALL be formed by concatenating (with a single space or newline — implementation choice, documented) pre-split pieces between consecutive split points.
**FR-7.2** Any final chunk whose token count exceeds `2 × chunk_size` SHALL be re-split via `RecursiveCharacterTextSplitter(chunk_size, recursive_overlap)` as a safety fallback; the fallback event SHALL be logged at `CHUNK` level.
**FR-7.3** The output of `LLMSemanticChunker.split` SHALL be a non-empty `list[str]` for any non-empty input text; an LLM producing zero splits for the entire document SHALL yield a single chunk containing the full document.

### FR-8 Pipeline integration
**FR-8.1** `QAGenerator.__init__` SHALL accept a `chunk_cfg: dict` parameter and construct `self.chunker = get_chunker(chunk_cfg, self.llm)`.
**FR-8.2** `QAGenerator.path2chunks` SHALL produce chunks by calling `self.chunker.split(raw_text)`, after the existing table and image stripping.
**FR-8.3** The chunk-output directory SHALL be renamed from `doc-chunks_{size}` to `doc-chunks_{size}_{method}` to prevent cross-method cache reuse.
**FR-8.4** `_nemo.py::main` SHALL read `cfg["chunking"]` and pass it into `QAGenerator`.

### FR-9 Prompt
**FR-9.1** A new prompt file `prompts/nemo_logical-chunk.txt` SHALL be added, accepting a single `{tagged_text}` input variable.
**FR-9.2** The prompt SHALL instruct the model to return JSON `{"split_after": [int, ...]}`.
**FR-9.3** Absence of the file when `method = "logical"` SHALL raise `FileNotFoundError` from `BaseLLM.read_prompt` (current behavior — no new handling needed).

### FR-10 Documentation
**FR-10.1** `CLAUDE.md` SHALL be updated to (a) list `nemo_logical-chunk` as a conditionally-required prompt, (b) describe the `[chunking]` section, (c) note the `RecursiveChunker` (batcher) vs `RecursiveTextChunker` (splitter) distinction.

---

## 4. Non-Functional Requirements

### NFR-1 Backward compatibility
An existing `cfg/nemo.toml` without `[chunking]` SHALL run the pipeline with behavior identical to the pre-change implementation (see FR-1.5, FR-1.3).

### NFR-2 Idempotency
`path2chunks` SHALL remain idempotent via its file-cache check. Re-running the same command without `--overwrite`-equivalent state SHALL incur zero LLM calls for chunking.

### NFR-3 Observability
Every logical-chunking window SHALL emit token/cost records through the existing `ChatResponse` decorator. Unusual conditions (parse failure, fallback re-split) SHALL be logged at the `CHUNK` loguru level.

### NFR-4 Determinism
`RecursiveTextChunker` SHALL be fully deterministic. `LLMSemanticChunker` SHALL be deterministic to the extent the underlying LLM is (typically when `temperature = 0.0`); this is a documented limitation, not a system requirement.

### NFR-5 Failure isolation
A failure in `LLMSemanticChunker.split` for a single document SHALL NOT corrupt or partially-write that document's chunks cache. Writes to `-chunks.json` happen only after successful assembly.

### NFR-6 Performance envelope
Logical chunking adds approximately `ceil(doc_tokens / (logical_presplit_tokens × logical_stride))` LLM calls per document at Stage 0.1. This is acceptable; no latency SLA is imposed.

---

## 5. Interfaces

### 5.1 Configuration interface
```toml
[chunking]
method = "recursive"          # enum: "recursive" | "logical"
chunk_size = 256              # int, target tokens per final chunk
recursive_overlap = 50        # int, overlap tokens (recursive) / pre-split size (logical)
logical_presplit_tokens = 50  # int, size of tagged pieces
logical_window = 40           # int, tagged pieces per LLM call
logical_stride = 30           # int, slide amount (<= logical_window)
```

### 5.2 Python interface
```python
# aisa/parse/chunkers.py
class Chunker(Protocol):
    def split(self, text: str) -> list[str]: ...

class RecursiveTextChunker(Chunker): ...
class LLMSemanticChunker(Chunker): ...

def get_chunker(chunk_cfg: dict, llm: BaseLLM | None = None) -> Chunker: ...
```

### 5.3 File interface
- **Input**: `{input_dir}/*.md`
- **Output**: `{root_dir}/doc-chunks_{chunk_size}_{method}/{doc_id}-chunks.json`
- **Schema** (unchanged):
  ```json
  {
    "doc_id": "...",
    "parsed_file": "...",
    "texts": [{"text": "...", "chunk_id": 0, "tokens": 123}, ...],
    "images": [...],
    "tables": [...]
  }
  ```

### 5.4 Prompt interface
`prompts/nemo_logical-chunk.txt`:
- Input variable: `{tagged_text}`
- Expected output: `{"split_after": [int, ...]}` (JSON object)

---

## 6. Acceptance Criteria

- **AC-1** With `method = "recursive"` and defaults matching today's `[embedding]` values, `-chunks.json` output MUST be identical to the pre-change output (spot-check one small and one large `.md`).
- **AC-2** With `method = "logical"` on a multi-topic markdown document, boundaries MUST align with topic shifts in at least one observed case (qualitative visual review during dev-test).
- **AC-3** Running `python _nemo.py --sdg` end-to-end under each method MUST produce a valid `full_sdg_output.json` consumable by `--prep`.
- **AC-4** Switching methods between runs MUST NOT cause stale chunks to be reused (directory naming verifies this).
- **AC-5** A malformed LLM response (forced via temperature manipulation or prompt corruption in a dev test) MUST NOT crash the pipeline; the affected window falls back cleanly.
- **AC-6** An existing checked-in `cfg/nemo.toml` (no `[chunking]` section) MUST run unchanged.

---

## 7. Risks and Open Questions

### 7.1 Risks
- **R-1** LLM misbehavior on structural edge cases (e.g., code blocks, tables reintroduced after markdown stripping gaps) could produce poor splits. Mitigation: the `chunk_size × 2` fallback in FR-7.2 caps damage.
- **R-2** Window overlap / stride mis-tuning could produce duplicated or missed boundaries. Mitigation: de-dup rule in FR-6.4.
- **R-3** Cost drift on large corpora. Mitigation: leave default `method = "recursive"`; operators opt in explicitly.

### 7.2 Open questions (must be resolved before implementation)
- **OQ-1** Final defaults for `logical_window`, `logical_stride`, `logical_presplit_tokens`.
- **OQ-2** Whether to use async `run_chain` (parallel windows) or sync `query` (sequential) for logical chunking. Parallel is faster but complicates split-point de-duplication across windows.
- **OQ-3** Whether to expose `overwrite` as a TOML flag as part of this feature (out of scope unless explicitly requested).
