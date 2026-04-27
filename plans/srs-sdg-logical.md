# Software Requirements Specification: SDG on Logical Chunks

**Feature:** Parallel SDG pipeline that generates per-logical-chunk contexts (Step 1: bundling only)
**Component:** `nvidia-sdq_custom`
**Version:** 0.1 (draft, Step 1)
**Status:** Proposed

---

## 1. Introduction

### 1.1 Purpose
This SRS defines requirements for a second, opt-in SDG flow in `_nemo.py` that operates on **logical chunks** instead of `RecursiveChunker`-packed bundles. The new flow runs in parallel to the existing `--sdg` flow and reuses no on-disk artifacts of it.

This document covers **Step 1 only** — bundle/context creation, ending with `-logic-ctx.json` written per document. Steps 2–4 (artifact extraction, QA generation, evaluation) are explicitly out of scope here and will be specified in follow-up SRS revisions.

### 1.2 Scope
In scope:
- New CLI flag `--sdg-logical`, new task key `sdg_logical`, new method `QAGenerator.run_sgd_logical_pipeline` in `_nemo.py`.
- New per-doc output file `{doc_id}-logic-ctx.json`.
- Naming convention reservation for the future per-stage outputs (`-logic-artifacts.json`, `-logic-qa_pairs.json`, `-logic-qa_eval.json`, `-logic-sdg.json`, `full_logic_sdg_output.json`).

Out of scope:
- The downstream stages themselves (artifacts/QA/eval) — placeholder names only.
- Changes to `--sdg` / `--prep` flow, including `run_sgd_pipeline`, `extract_artifacts`, `generate_qa_pairs`, `evaluate_qa_pairs`, `run_data_prep_pipeline`.
- Changes to `aisa/parse/chunkers.py` or any chunker class. The new flow consumes whatever `path2chunks` produces.
- Changes to prompts, `cfg/nemo.toml`, or `aisa/gen/*`.
- Changes to deprecated code under `_depr/`.

### 1.3 Definitions
- **Logical chunk** — an output element of `LLMSemanticChunker` (mode `logical`) or `HybridLogicalChunker` (mode `random_logical`), grouped by an LLM into a semantically coherent unit.
- **Recursive bundle** — what `aisa/parse/chunk.py::RecursiveChunker` (the LLM-input batcher, **not** a text splitter) produces in the existing flow: a list of consecutive chunks packed under a token budget.
- **Bundle / context entry** — one element of `-ctx.json` (existing) or `-logic-ctx.json` (new): `{chunks: [...], tokens: int}`.
- **Logical flow** — the new pipeline introduced by this SRS, gated on `--sdg-logical`.
- **Bundled flow** — the existing pipeline, gated on `--sdg`.

### 1.4 References
- `plans/plan-sdg-logical.md` — companion implementation plan.
- `plans/srs-logical-chunking.md` — SRS for the chunker family that produces the logical chunks consumed here.
- `CLAUDE.md` — project conventions and architecture.

---

## 2. Overall Description

### 2.1 Product Perspective
`_nemo.py` exposes three tasks today (`chunk`, `sdg`, `prep`). This feature adds a fourth (`sdg_logical`) that shares Stage 0.1 (`path2chunks`) with the existing flow but replaces the bundling stage of `extract_artifacts` with a 1-to-1 mapping (one logical chunk → one bundle). All later stages — when implemented in follow-up steps — will be parallel to, not shared with, the bundled flow.

### 2.2 User Classes
- **Pipeline operator** — runs `python _nemo.py --sdg-logical` to inspect the per-logical-chunk bundling output for debugging or qualitative comparison against the bundled flow.
- **Pipeline developer** — extends the logical flow with subsequent stages (Steps 2–4).

### 2.3 Operating Environment
Identical to the bundled flow: Python 3.x, dependencies in `reqs.txt`, `[llm]` provider configured, `[chunking]` configured for one of the LLM-driven methods.

### 2.4 Constraints
- All chunk and bundle schemas MUST be identical to those of the bundled flow so that future Steps 2–4 can re-use existing call sites with only filename swaps.
- File-cache idempotency MUST be preserved (`if Path(out).exists() and not self.overwrite`).
- Filename derivation MUST go through `self.doc_paths[file_path].replace("-chunks.json", "-logic-<suffix>.json")`. This is safe because `self.doc_paths[file_path]` is always set to `{doc_id}-chunks.json` at `_nemo.py:177`, regardless of the active chunking method.
- Token counting (when used for warnings) MUST use the existing `get_token_count` helper at `_nemo.py:42` (tiktoken `gpt-3.5-turbo` encoding).
- The new flow MUST NOT modify, read, or invalidate any artifact written by the bundled flow.

### 2.5 Assumptions
- The user has set `[chunking].method` to `"logical"` or `"random_logical"` before invoking `--sdg-logical`. The chunker classes themselves are already specified and verified by `srs-logical-chunking.md`.
- `path2chunks` already writes `{doc_id}-chunks.json` (and, in mode `random_logical`, `{doc_id}-logic-chunks.json`) and returns the appropriate chunk list per mode.
- The user maintains `[llm].max_input_tokens` consistent with downstream stage expectations (the warning emitted in FR-4 is informational, not enforcing).

---

## 3. Functional Requirements

### FR-1 CLI surface
**FR-1.1** `_nemo.py` SHALL accept a new boolean flag `--sdg-logical` parsed by argparse. The dest attribute SHALL be `sdg_logical` (argparse default for the dashed flag).
**FR-1.2** `_nemo.py::__main__` SHALL include `"sdg_logical": args.sdg_logical` in the synthesised `cfg["nemo_task"]` dict.
**FR-1.3** The "no task selected" guard at `_nemo.py:604` SHALL include `--sdg-logical` in the set that satisfies the requirement.
**FR-1.4** `--sdg-logical` SHALL be combinable with `--chunk-only`, `--sdg`, and `--prep` in the same invocation (per the existing `for task_name, should_run in cfg["nemo_task"].items()` dispatch).

### FR-2 Task registration
**FR-2.1** `QAGenerator.tasks` SHALL register a new entry `"sdg_logical": self.run_sgd_logical_pipeline`.
**FR-2.2** Task name ordering in `self.tasks` SHALL place `sdg_logical` after `sdg` (cosmetic; aids readability of the dispatch loop).

### FR-3 Chunking-method validation
**FR-3.1** `run_sgd_logical_pipeline` SHALL, before iterating documents, verify that `self.chunk_cfg.get("method")` is in `{"logical", "random_logical"}`.
**FR-3.2** On violation, the system SHALL raise `ValueError` whose message names the offending value and the allowed set.
**FR-3.3** Validation SHALL fail fast: no per-document work is performed when the method is invalid.

### FR-4 Per-document context construction
**FR-4.1** For each `*.md` in `self.input_dir`, the system SHALL invoke `self.path2chunks(file_path)` to obtain the chunk list. No re-implementation of chunking is permitted.
**FR-4.2** The system SHALL build the bundle list as follows:
```python
ctx = [{"chunks": [chunk], "tokens": chunk.get("tokens", 0)} for chunk in chunks]
```
i.e., one bundle per logical chunk, single-element `chunks` list, top-level `tokens` equal to that chunk's `tokens`.
**FR-4.3** The system SHALL NOT invoke `aisa/parse/chunk.py::RecursiveChunker` in this flow.
**FR-4.4** The system SHALL NOT invoke `_trim_overlap_for_context` in this flow.
**FR-4.5** For any bundle whose `tokens > self.llm.cfg.max_input_tokens`, the system SHALL emit one `loguru` log line at level `"CHUNK"` naming the document file name, the chunk's `chunk_id`, the bundle's token count, and the configured budget. Processing SHALL continue.

### FR-5 Output file
**FR-5.1** The output filename SHALL be derived as `self.doc_paths[file_path].replace("-chunks.json", "-logic-ctx.json")`.
**FR-5.2** The file SHALL contain the bundle list serialised via `aisa.utils.files.write_json` (i.e., a top-level JSON array).
**FR-5.3** Each array element SHALL conform to the schema `{"chunks": list[Chunk], "tokens": int}` where `Chunk` matches the chunk schema written by `path2chunks` (`{"text": str, "chunk_id": int, "tokens": int}`, plus `source_chunk_ids: list[int]` in mode `random_logical`).
**FR-5.4** The output file SHALL live in the same directory as the corresponding `-chunks.json` (i.e., `self.chunk_dir = doc-chunks_{size}_{method}`). No new directory is created by this feature.

### FR-6 Idempotency
**FR-6.1** Before computing bundles, the system SHALL check `Path(out_path).exists() and not self.overwrite`. On hit, it SHALL `return files.read_json(out_path)` and emit a single `"CHUNK"` log line to that effect.
**FR-6.2** A re-run with no input changes and `self.overwrite = False` SHALL incur zero file writes for `-logic-ctx.json`.

### FR-7 Failure isolation
**FR-7.1** A failure during context construction for one document SHALL NOT corrupt or partially-write that document's `-logic-ctx.json` (write happens after the in-memory list is fully built).
**FR-7.2** A failure for document A SHALL NOT abort processing of document B unless the underlying error is the FR-3 method-validation error (which is raised before the loop).

### FR-8 Reserved naming
**FR-8.1** The following filenames are RESERVED by this feature for future Step 2–4 implementations and SHALL NOT be used by any other flow:
- `{doc_id}-logic-artifacts.json`
- `{doc_id}-logic-qa_pairs.json`
- `{doc_id}-logic-qa_eval.json`
- `{doc_id}-logic-sdg.json`
- `{root_dir}/full_logic_sdg_output.json`

**FR-8.2** This SRS does not define their schema; doing so is the responsibility of the SRS revisions covering Steps 2–4.

### FR-9 Non-interference with bundled flow
**FR-9.1** Running `--sdg-logical` SHALL NOT read or write any of: `{doc_id}-ctx.json`, `{doc_id}-artifacts.json`, `{doc_id}-qa_pairs.json`, `{doc_id}-qa_eval.json`, `{doc_id}-sdg.json`, `{root_dir}/full_sdg_output.json`.
**FR-9.2** Running `--sdg` SHALL NOT read or write any file enumerated in FR-8.1 or `{doc_id}-logic-ctx.json`.
**FR-9.3** Both flows MAY share the same `{doc_id}-chunks.json` (and, in mode `random_logical`, `{doc_id}-logic-chunks.json`), as those are inputs to both.

### FR-10 Termination
**FR-10.1** `run_sgd_logical_pipeline` SHALL return after writing the last `-logic-ctx.json`. No further LLM calls SHALL be made by this method.
**FR-10.2** This is intentional debug-time behavior; future SRS revisions covering Steps 2–4 will extend the method, not replace it.

---

## 4. Non-Functional Requirements

### NFR-1 Backward compatibility
Configurations and invocations that do not pass `--sdg-logical` SHALL behave identically to the pre-change implementation. No default value, schema, or path used by `--sdg`/`--chunk-only`/`--prep` is altered by this feature.

### NFR-2 Observability
Each per-document run SHALL emit at least one `"CHUNK"` log line: either the count of bundles written or the cache-hit notice (FR-6). Oversized-bundle warnings (FR-4.5) SHALL also use the `"CHUNK"` level.

### NFR-3 Determinism
`run_sgd_logical_pipeline` SHALL be fully deterministic given a fixed `path2chunks` output. No randomness, no LLM calls inside this method.

### NFR-4 Performance envelope
The new method's only non-trivial cost is the `path2chunks` call, which is itself cached on `-chunks.json` / `-logic-chunks.json`. Bundle construction is O(N) in the number of logical chunks per doc and O(1) per chunk. No latency SLA is imposed.

### NFR-5 Schema parity
The shape of `-logic-ctx.json` SHALL be a strict subset of `-ctx.json`'s shape (single-element `chunks` lists, but otherwise identical fields and types) so that Step 2's future implementation can substitute one for the other with only a filename change.

---

## 5. Interfaces

### 5.1 CLI interface
```text
python _nemo.py --sdg-logical [--sdg] [--prep] [--chunk-only]
                [--cfg cfg/nemo.toml] [--input_dir DIR] [--output_dir DIR]
```
- Requires `[chunking].method ∈ {"logical", "random_logical"}` in the resolved config.
- Requires at least one of `--chunk-only`, `--sdg`, `--prep`, `--sdg-logical`.

### 5.2 Python interface
```python
class QAGenerator:
    async def run_sgd_logical_pipeline(self) -> None: ...
    def _build_logical_contexts(
        self, file_path: Path, chunks: dictlist
    ) -> dictlist: ...
```
`self.tasks["sdg_logical"]` is the registered entry point.

### 5.3 File interface
- **Input** (per doc): `{self.chunk_dir}/{doc_id}-chunks.json` (always), plus `{self.chunk_dir}/{doc_id}-logic-chunks.json` (mode `random_logical` only). These are produced by `path2chunks`; this feature does not write them.
- **Output** (per doc): `{self.chunk_dir}/{doc_id}-logic-ctx.json`.
- **Schema**:
  ```json
  [
    {
      "chunks": [
        {"text": "...", "chunk_id": 0, "tokens": 123}
      ],
      "tokens": 123
    },
    ...
  ]
  ```
  In mode `random_logical`, each chunk additionally carries `source_chunk_ids: list[int]` (passed through verbatim from `-logic-chunks.json`).

### 5.4 Configuration interface
No new configuration keys. The feature reads:
- `cfg["chunking"]["method"]` (validated against the allow-list).
- `cfg["llm"]["max_input_tokens"]` (used only for the FR-4.5 warning threshold).
- `cfg["general"]["data_dir"]`, `cfg["general"]["output_dir"]` (existing).

---

## 6. Acceptance Criteria

- **AC-1** With `[chunking].method = "random_logical"` on a small `*.md` set, `python _nemo.py --sdg-logical` writes `{doc_id}-logic-ctx.json` for each doc. The file's array length equals the `texts` array length in the matching `{doc_id}-logic-chunks.json`.
- **AC-2** Each `-logic-ctx.json` entry has a single-element `chunks` list whose `chunk_id`, `text`, and `tokens` are byte-equal to the corresponding entry in `-logic-chunks.json` (or `-chunks.json` for mode `logical`).
- **AC-3** With `[chunking].method = "recursive"`, `python _nemo.py --sdg-logical` raises `ValueError` before any document is processed; the error message names `"recursive"` and the allowed methods.
- **AC-4** Running `--sdg-logical` immediately after a `--sdg` run does not modify `-ctx.json`, `-artifacts.json`, `-qa_pairs.json`, `-qa_eval.json`, `-sdg.json`, or `full_sdg_output.json`. Running `--sdg` after `--sdg-logical` does not modify `-logic-ctx.json`.
- **AC-5** A second invocation of `--sdg-logical` with unchanged inputs and `self.overwrite = False` writes nothing (verified via `mtime` or hash comparison).
- **AC-6** With `[llm].max_input_tokens` set artificially below the largest logical chunk's token count, the run emits at least one `"CHUNK"` warning naming the offending `chunk_id` and tokens, and completes without raising.
- **AC-7** Combining flags (`--sdg-logical --chunk-only` or `--sdg-logical --sdg`) runs both pipelines in a single invocation, in the order they appear in `cfg["nemo_task"]`.

---

## 7. Risks and Open Questions

### 7.1 Risks
- **R-1** Single-logical-chunk contexts may produce shallower QA pairs than multi-segment bundles when Step 3 is wired (the bundled flow's prompt explicitly demands cross-segment reasoning). Step 1 cannot detect this — it only writes the inputs. Mitigation: Step 3 SRS will revisit the prompt.
- **R-2** Token-budget overflows (FR-4.5) become enforcement decisions in Step 2. Logging-only here is the right Step-1 behavior but defers the question. Mitigation: explicit FR-4.5 warning ensures the operator sees the issue before reaching Step 2.
- **R-3** Filename collision risk if a future, unrelated feature also adopts a `-logic-` prefix. Mitigation: FR-8.1 reserves the names enumerated; future features must avoid them.

### 7.2 Open questions (non-blocking for Step 1)
- **OQ-1** Whether `--sdg-logical` should later gain a sibling `--prep-logical`, or whether `--prep` should learn to consume `full_logic_sdg_output.json` via a flag. Defer to data-prep planning.
- **OQ-2** Whether to support running `--sdg` and `--sdg-logical` in the same process under different `[chunking]` configs. Currently impossible because the chunker is built once at `QAGenerator.__init__`. Re-evaluate when a unified comparison report is wanted.
- **OQ-3** Whether to surface the bundle count in the per-doc log line, in `MODEL`/`COST` log levels, or both. Cosmetic; defer.
