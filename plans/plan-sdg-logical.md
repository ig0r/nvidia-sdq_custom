# Plan: SDG Pipeline on Logical Chunks

A second, parallel SDG flow that generates QA pairs **per logical chunk** instead of per `RecursiveChunker`-bundled context. Selected via a new CLI flag, runs alongside (not in place of) the existing `--sdg` flow.

This plan covers **Step 1 only** — bundling/context creation. Steps 2–4 (artifact extraction, QA generation, evaluation) are deferred and will be planned separately once Step 1 is verified.

## Scope of Step 1

- One **logical chunk = one bundle = one context entry**. No `RecursiveChunker` packing, no overlap trimming.
- Pipeline **stops after writing `-logic-ctx.json`**. No LLM calls beyond what `path2chunks` itself triggers (in `logical` / `random_logical` modes).
- Goal: a debuggable artifact showing exactly what the downstream QA-gen call will see when wired in.

## Naming convention

Both flows must coexist in a single output directory without cache collisions. The new flow gets a `-logic-` infix on every per-stage suffix:

| Stage | Existing flow | Logical flow |
|---|---|---|
| Chunk inputs (already written by `path2chunks`) | `-chunks.json` | `-chunks.json` (mode `logical`) / `-logic-chunks.json` (mode `random_logical`) |
| Bundles | `-ctx.json` | `-logic-ctx.json` |
| Artifacts (Step 2, future) | `-artifacts.json` | `-logic-artifacts.json` |
| QA pairs (Step 3, future) | `-qa_pairs.json` | `-logic-qa_pairs.json` |
| Eval (Step 4, future) | `-qa_eval.json` | `-logic-qa_eval.json` |
| Per-doc combined (future) | `-sdg.json` | `-logic-sdg.json` |
| Corpus aggregate (future) | `full_sdg_output.json` | `full_logic_sdg_output.json` |

Naming derivation works because `self.doc_paths[file_path]` always points at `{doc_id}-chunks.json` (`_nemo.py:177`) regardless of chunking method, so a `.replace("-chunks.json", "-logic-ctx.json")` substitution is safe in all modes.

## Design choices

1. **New task**, new method, new flag. Existing `--sdg` stays byte-for-byte unchanged.
   - Task key: `sdg_logical` (snake_case, matches existing `chunk` / `sdg` / `prep` keys).
   - CLI flag: `--sdg-logical` (argparse maps to `args.sdg_logical`).
   - Method: `QAGenerator.run_sgd_logical_pipeline`.

2. **Required chunking mode**: `[chunking].method` must be `"logical"` or `"random_logical"`. `"recursive"` is rejected with a clear `ValueError` — the new flow is meaningless without semantically grouped chunks.

3. **Bundling = identity**: each logical chunk becomes one bundle of one chunk. No `RecursiveChunker`. No `_trim_overlap_for_context` (logical chunks have no sliding overlap to remove — `HybridLogicalChunker` already trims overlap during assembly, and `LLMSemanticChunker` pre-splits with `chunk_overlap=0`).

4. **Output schema** for `-logic-ctx.json`: identical shape to the existing `-ctx.json`.
   ```json
   [
     {"chunks": [{"chunk_id": 0, "text": "...", "tokens": 123}], "tokens": 123},
     {"chunks": [{"chunk_id": 1, "text": "...", "tokens": 456}], "tokens": 456}
   ]
   ```
   Single-element `chunks` list per entry. Schema parity is the contract that lets future steps reuse `extract_artifacts` / `generate_qa_pairs` plumbing with a one-line filename swap.

5. **Token-budget**: a single logical chunk can in principle exceed `[llm].max_input_tokens` (e.g. mode 3's de facto cap is `hybrid_window × chunk_size`, which can be tuned above the LLM input budget for downstream stages). For Step 1, **log a `CHUNK`-level warning per oversized bundle and pass through**. Hard decisions (split? truncate? reject?) are deferred to when Steps 2–4 wire the LLM calls.

6. **Idempotency**: skip writing `-logic-ctx.json` if it exists and `self.overwrite` is `False`. Same convention as every other stage in the file.

## Concrete changes to `_nemo.py`

### New method `_build_logical_contexts(file_path, chunks) -> dictlist`
- Compute `out_path = self.doc_paths[file_path].replace("-chunks.json", "-logic-ctx.json")`.
- If cached and not `self.overwrite`: `return files.read_json(out_path)`.
- Build:
  ```python
  ctx = [{"chunks": [chunk], "tokens": chunk.get("tokens", 0)} for chunk in chunks]
  ```
- For each entry where `entry["tokens"] > self.llm.cfg.max_input_tokens`: emit `logger.log("CHUNK", ...)` warning naming `doc_id`, `chunk_id`, token count, and the budget.
- `files.write_json(ctx, out_path)`.
- Return `ctx`.

### New method `run_sgd_logical_pipeline()`
- Validate `self.chunk_cfg.get("method") in {"logical", "random_logical"}`; otherwise raise `ValueError` naming the offending method and the allowed set.
- For each `*.md` in `self.input_dir`:
  - `chunks = self.path2chunks(file_path)`
  - `ctx = self._build_logical_contexts(file_path, chunks)`
  - `logger.log("CHUNK", f"{file_path.name}: {len(ctx)} logical-context entries -> {out_path}")`
- **Stop.** No `extract_artifacts`, no `generate_qa_pairs`, no `evaluate_qa_pairs`.

### Task registration (`_nemo.py:155`)
Add `"sdg_logical": self.run_sgd_logical_pipeline` to `self.tasks`.

### CLI / `__main__` (`_nemo.py:584`+)
- Add `parser.add_argument("--sdg-logical", action="store_true", help="Run SDG on logical chunks (Step 1: bundle only)")`.
- Add `"sdg_logical": args.sdg_logical` to the `nemo_task` dict.
- Include in the "no task selected" error check.

### Documentation
- `CLAUDE.md`: add a one-liner under "Entry Point & Commands" for `--sdg-logical` once Step 1 is verified working. Skip for now.

## Out of scope (deferred to future steps)

- **Step 2** — Logical-chunk artifact extraction. Likely 1 LLM call per logical chunk via `nemo_artifacts`, output to `-logic-artifacts.json`. To plan separately.
- **Step 3** — Logical-chunk QA generation. `nemo_qa-gen` was written for multi-segment bundles; with one segment per call, the "connect multiple separated segments", `multi_hop` / `min_hops` / `max_hops` / `hop_contexts` machinery no longer applies. A new prompt variant is likely needed. To plan separately.
- **Step 4** — Eval. Probably reusable as-is; defer the call until Step 3 lands.
- **Data prep**: `run_data_prep_pipeline` consumes `full_sdg_output.json`. A logical-flow equivalent would consume `full_logic_sdg_output.json`. Either share `--prep` via a flag or add `--prep-logical`. Deferred.

## Verification (Step 1 acceptance)

1. With `[chunking].method = "random_logical"` on a small `*.md` set:
   - Run `python _nemo.py --sdg-logical --cfg cfg/nemo.toml`.
   - Confirm `-logic-ctx.json` written for each doc, alongside the existing `-chunks.json` and `-logic-chunks.json`.
   - Confirm the entry count equals the count of logical chunks in `-logic-chunks.json`.
   - Confirm each entry has a single-element `chunks` list with the correct `chunk_id` and `text` (byte-equal to `-logic-chunks.json` `texts[i]`).
2. Same with `[chunking].method = "logical"` (no `-logic-chunks.json` exists; entry count matches `-chunks.json`).
3. With `[chunking].method = "recursive"`: confirm `--sdg-logical` raises `ValueError` immediately.
4. Re-run: confirm cache hit (no rewrite, no LLM calls beyond chunking which itself is cached).
5. Spot-check oversized-chunk warning by setting `[llm].max_input_tokens` artificially low and confirming the `CHUNK` log line fires per oversized entry without crashing.

## Open question (deferred until needed)

The existing `--sdg` and the new `--sdg-logical` cannot run in the same process invocation against the same chunking config, because both call `path2chunks` and the `[chunking].method` decision is made once at `QAGenerator.__init__`. This is fine for Step 1 (the user picks one mode per run), but it means a corpus that wants "both flows side-by-side" needs two invocations with different `[chunking]` settings. Not blocking. Re-evaluate if/when a unified report is wanted.
