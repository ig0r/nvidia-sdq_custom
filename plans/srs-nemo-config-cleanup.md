# Software Requirements Specification: Conservative Config Cleanup for `cfg/nemo.toml` + `cfg/nemo_specs.toml`

**Feature:** Trim dead sibling-project carryover sections/keys from the two `_nemo.py` TOML
configs (`cfg/nemo.toml`, `cfg/nemo_specs.toml`) under a **conservative** scope (remove only
top-level items that **no script in this repo reads at all**), keeping both files multi-mode
capable, with **zero source-code change** and **zero behavior change** to any pipeline mode.
**Component:** `nvidia-sdq_custom`
**Version:** 0.1
**Status:** Implemented
**Companion plan:** `plans/plan-nemo-config-cleanup.md`
**Working notes (background, not authoritative):**
`/Users/igor/.claude/plans/perform-research-what-settings-elegant-hartmanis.md`

---

## 1. Introduction

### 1.1 Purpose
`cfg/nemo.toml` and `cfg/nemo_specs.toml` carry a large block of configuration inherited from a
larger sibling project (`[pub242]`, `[jsa]`, `[nlp]`, `[doc]`, `[general.theme]`,
`[general.tasks]`, `[general].metadata_folder`). A completed audit (§2) confirms **no script in
this repo reads any of it**. This SRS specifies the requirements to delete exactly those dead
top-level items from **both** files identically, add a one-line provenance header to each, and
prove — by `read_toml` key assertion plus a 1-doc chunk smoke — that every actively-consumed
section is preserved and no pipeline mode changes behavior. It specifies a config-only change;
**no `.py` file is touched**.

### 1.2 Scope

**In scope**
- Delete, from **both** `cfg/nemo.toml` and `cfg/nemo_specs.toml` (identical 143-line layout),
  the top-level items the audit (§2) proves are read by no script: `[general].metadata_folder`,
  `[general.theme]`, `[general.tasks]`, `[pub242]` (+ `.tasks`, `.saliency`), `[jsa]` (+
  `.tasks`), `[nlp]` (+ `.tasks`), `[doc]` (+ `.rag`, `.mlp`, `.tasks`).
- Preserve, byte-for-byte except for the deletions, every actively-consumed section: `[general]`
  (reduced to `output_dir`, `data_dir`), `[llm]`, `[embedding]`, `[chunking]`, `[langextract]`.
- Add a one-line `#` header comment to each file recording that it is consumed only by
  `_nemo.py` (plus `extract_artifacts.py` for `[langextract]` when explicitly handed
  `cfg/nemo.toml`), so future cruft is not silently re-added.
- A `read_toml` top-level-key assertion plus a 1-doc `--chunk-only` smoke as acceptance.

**Out of scope**
- Any `.py` change (this is the load-bearing safety property — FR-5).
- **Moderate** trimming: removing keys *inside* actively-unpacked sections that no mode
  functionally consumes (`[llm].chunk_size`, `[llm].max_chain_tokens`, `[embedding].prefix`,
  the `logical_*` keys when only `random_logical` is ever run, the `[embedding].chunk_size`/
  `recursive_overlap` fallback duplicates). Explicitly deferred — see §10.
- **Aggressive** trimming: collapsing the two files into one, or stripping multi-mode keys to
  match only the `random_logical` route. Explicitly deferred — see §10.
- Editing the four `_specs` configs (`extract_artifacts_specs.toml`, `generate-qa_specs.toml`,
  `self-check/self-check-qa_specs.toml`) or any other TOML. Only the two `cfg/*.toml` change.
- Removing `[langextract]` from `cfg/nemo_specs.toml` even though it is a dead copy there (§2) —
  kept for cross-file parity under conservative scope; recorded as a finding only.
- Renaming, reformatting, or re-commenting kept keys beyond the single header line.

---

## 2. Background / Current State

`cfg/nemo.toml` is `_nemo.py`'s default config (`--cfg ./cfg/nemo.toml`); `cfg/nemo_specs.toml`
is the explicit-`--cfg` variant used by the Ollama/specs route
(`plans/srs-ollama-random-logical-pipeline.md`). Both files are an **identical 143-line layout**
(verified: only `[general].output_dir`/`data_dir`, `[llm].model`, and
`[chunking].relevance_concurrency` differ in *values*; section structure is line-for-line
identical). `_nemo.py` is the only consumer for most sections; `extract_artifacts.py` *can* also
be pointed at `cfg/nemo.toml` and then reads `[general].output_dir`, `[chunking].method`, and
`[langextract]`. `generate-qa.py` and `self-check/self-check-qa.py` never read either file.

Audit of every consumer (verified against the working tree at the cited loci):

| Locus (file:line) | Binding | Disposition |
|---|---|---|
| `_nemo.py:840` | `root_dir = cfg["general"]["output_dir"]` — explicit key, **not** a whole-`[general]` unpack | Keep `[general].output_dir` |
| `_nemo.py:841` | `input_dir = cfg["general"]["data_dir"]` — explicit key | Keep `[general].data_dir` |
| `_nemo.py:842` | `BaseLLM(LLMConfig(**cfg["llm"]))` — whole `[llm]` Pydantic-unpacked | Keep `[llm]` whole (FR-2) |
| `_nemo.py:843` | `Embedder(EmbedConfig(**cfg["embedding"]))` — whole `[embedding]` Pydantic-unpacked | Keep `[embedding]` whole (FR-2) |
| `_nemo.py:844` | `chunk_cfg = cfg.get("chunking")` | Keep `[chunking]` whole |
| `_nemo.py:868-879` | `global_cfg["nemo_task"]` synthesised from CLI flags; `global_cfg["general"]["output_dir"]`/`["data_dir"]` set by `--output_dir`/`--input_dir` — explicit keys | Keep `[general].output_dir`/`data_dir` |
| `_nemo.py:172-182` | `chunk_cfg.get("relevance_concurrency"/"relevance_model"/"relevance_filter"/"method")` | Keep `[chunking]` keys |
| `aisa/parse/chunkers.py:304-329` | `get_chunker` reads `method`, `chunk_size`, `recursive_overlap`, `logical_presplit_tokens`/`logical_window`/`logical_stride` (317-319), `hybrid_window`/`hybrid_stride` (328-329) via `.get(...)` | Keep `[chunking]` keys (multi-mode) |
| `aisa/parse/chunkers.py:265-269` | `hybrid_window * chunk_size <= llm.cfg.max_input_tokens` validation | Keep `[chunking]`/`[llm]` |
| `extract_artifacts.py:934` | `cfg.get("general", {}).get("output_dir")` (only when handed `cfg/nemo.toml`) | Keep `[general].output_dir` |
| `extract_artifacts.py:955-960` | reads `[chunking].method`; enforces `== "random_logical"` | Keep `[chunking].method` |
| `extract_artifacts.py:962-964` | **raises `KeyError`** if `[langextract]` present without `[artifact_extraction]` | See §8 R3 — `cfg/nemo_specs.toml` MUST NOT be passed to `extract_artifacts.py` |
| `aisa/utils/files.py:91-93` | `read_toml` = `tomllib.load` (top-level keys = TOML tables) | Drives §9 key assertion |
| `_nemo.py` / `extract_artifacts.py` / `aisa/` (whole tree) | grep for `metadata_folder`, `general.theme`, `[general].tasks`, `pub242`, `jsa`, `nlp`-table, `doc`-table, `saliency`, `.mlp`, `doc.rag` → **zero matches** | All DELETE-listed items are dead (FR-1) |

Verified audit facts (spot-checked this session against the working tree):
- `_nemo.py` accesses `[general]` **only** by explicit sub-key (`["output_dir"]`,
  `["data_dir"]`) at `:840`, `:841`, `:868`, `:870`, `:879` — never `cfg["general"]` whole, and
  never any of `metadata_folder`/`theme`/`tasks`. Deleting unread `[general]` sub-tables/keys is
  therefore inert.
- `[llm]` and `[embedding]` ARE whole-section Pydantic-unpacked (`LLMConfig(**…)`,
  `EmbedConfig(**…)` at `_nemo.py:842-843`). Their *keys* are out of conservative scope and
  MUST NOT be touched (an unexpected/missing key risks a Pydantic `TypeError`/validation error).
- A repo-wide grep across `_nemo.py`, `extract_artifacts.py`, and `aisa/` for every
  DELETE-listed identifier returns **no matches** — confirming none of the carryover sections
  is read by any code path of either pipeline route.
- `cfg/nemo.toml` and `cfg/nemo_specs.toml` are structurally identical (same 143-line section
  layout; the DELETE line ranges in §3 apply verbatim to both).
- `[langextract]` in `cfg/nemo_specs.toml` is a **dead copy**: the specs/Ollama route points
  `extract_artifacts.py` at `extract_artifacts_specs.toml`, never at `cfg/nemo_specs.toml`
  (and doing so would `KeyError` at `extract_artifacts.py:962-964`). In `cfg/nemo.toml`,
  `[langextract]` is live only via `extract_artifacts.py --cfg cfg/nemo.toml`. Under
  conservative scope it is **kept in both** for parity; recorded here as a finding.
- Redundancy findings (kept — out of conservative scope, recorded for §10): the `logical_*`
  keys are dead for `random_logical` but live for `method="logical"` (`--sdg`/`--chunk-only`);
  `[embedding].chunk_size`/`recursive_overlap` are a fallback-only duplicate of `[chunking]`
  (live for `--prep`/the `[chunking]`-absent fallback); `[llm].chunk_size=20`,
  `[llm].max_chain_tokens`, `[embedding].prefix` are accepted by the Pydantic configs but not
  functionally consumed by any mode.

---

## 3. Functional Requirements

### FR-1 — Delete the dead carryover items (identical edit to both files)
- **FR-1.1** The following top-level items SHALL be deleted **in full** (table header,
  sub-tables, keys, and their inline comments) from **`cfg/nemo.toml`**, at the current line
  ranges:
  - `[general].metadata_folder` (line 5)
  - `[general.theme]` (lines 6–9)
  - `[general.tasks]` (lines 10–14)
  - `[pub242]` + `[pub242.tasks]` + `[pub242.saliency]` (lines 16–40)
  - `[jsa]` + `[jsa.tasks]` (lines 87–99)
  - `[nlp]` + `[nlp.tasks]` (lines 101–110)
  - `[doc]` + `[doc.rag]` + `[doc.mlp]` + `[doc.tasks]` (lines 112–143)
- **FR-1.2** The **identical structural deletion** (same items, same effective result) SHALL be
  applied to **`cfg/nemo_specs.toml`** (same 143-line layout; same line ranges).
- **FR-1.3** No item NOT in the FR-1.1 list SHALL be deleted. In particular `[langextract]`
  SHALL be retained in **both** files (including its dead-copy presence in `cfg/nemo_specs.toml`
  — §2).
- **FR-1.4** Surrounding whitespace MAY be normalized so the result is a clean, contiguous file
  (no leading blank line, single trailing newline); kept sections SHALL NOT be reordered or
  reformatted beyond removing the deleted blocks.

### FR-2 — Kept-set invariant
- **FR-2.1** After the edit, the set of top-level TOML tables in **each** file
  (`sorted(read_toml(...).keys())`) SHALL be **exactly**
  `['chunking', 'embedding', 'general', 'langextract', 'llm']`.
- **FR-2.2** `[general]` SHALL retain exactly its two consumed keys, `output_dir` and
  `data_dir`, with their current values and inline value-history comments unchanged.
- **FR-2.3** `[llm]`, `[embedding]`, `[chunking]`, and `[langextract]` SHALL retain **all**
  their keys and values byte-for-byte (the Pydantic-unpacked `[llm]`/`[embedding]` are
  especially sensitive — no key added, removed, renamed, or reordered).
- **FR-2.4** The value-level differences that distinguish the two files SHALL be preserved:
  `cfg/nemo.toml` `[general].output_dir`/`data_dir`, `[llm].model = "gpt-4o-mini"`,
  `[chunking].relevance_concurrency = 8`; `cfg/nemo_specs.toml`
  `[general].output_dir = "./data/specs_20260516"`, `data_dir = "./rawdata-pubs/parsed-specs"`,
  `[llm].model = "gpt-oss:120b"#…`, `[chunking].relevance_concurrency = 2`. The cleanup SHALL
  NOT homogenize these.

### FR-3 — Provenance header comment
- **FR-3.1** Each file SHALL gain a one-line `#` comment at the very top recording that the
  file is consumed only by `_nemo.py` (and by `extract_artifacts.py` for `[langextract]` when
  it is explicitly handed `cfg/nemo.toml`), so future carryover cruft is not re-added.
- **FR-3.2** The header SHALL be a TOML comment (`#`-prefixed) and SHALL NOT introduce any key
  or table; it MUST NOT change FR-2.1's parsed top-level-key set.

### FR-4 — Cross-file parity
- **FR-4.1** After the edit, the two files SHALL differ **only** in the value-level rows
  enumerated in FR-2.4 (and the existing inline value-history comments). The section/key
  *structure* SHALL be identical between the two files, exactly as it is today.
- **FR-4.2** A structural diff that ignores the FR-2.4 value rows SHALL show no
  section/key-level difference between the two files.

### FR-5 — No code change / no behavior change (safety invariant)
- **FR-5.1** No `.py` file SHALL be modified by this feature. `git diff --stat` for the change
  SHALL list **only** `cfg/nemo.toml` and `cfg/nemo_specs.toml` (plus the two new `plans/*.md`).
- **FR-5.2** Every `_nemo.py` mode (`--chunk-only`, `--sdg`, `--sdg-logical`, `--prep`) and
  every `[chunking].method` (`recursive`, `logical`, `random_logical`) SHALL behave identically
  before and after the edit — guaranteed structurally because `_nemo.py` reads `[general]` by
  explicit sub-key (§2) and the deleted tables are referenced by no code path (FR-1 / §2 grep).
- **FR-5.3** `extract_artifacts.py --cfg cfg/nemo.toml` SHALL continue to resolve
  `[general].output_dir`, `[chunking].method`, and `[langextract]` exactly as before.
- **FR-5.4** The Pydantic unpack of `[llm]`/`[embedding]` (`_nemo.py:842-843`) SHALL be
  unaffected: no key in those sections is added/removed (FR-2.3), so no new `TypeError`/
  validation path is introduced.

---

## 4. External Interfaces

- **`aisa/utils/files.read_toml(path)`** (`aisa/utils/files.py:91-93`, `tomllib.load`) — the
  parser whose top-level dict keys define the FR-2.1 invariant; used by the §9 assertion.
- **`_nemo.py` CLI** — `--cfg` (default `./cfg/nemo.toml`), `--input_dir`, `--output_dir`,
  `--chunk-only`; the §9 smoke uses these (override input + output, no toml edit).
- **`extract_artifacts.py` CLI** — `--cfg` may be `cfg/nemo.toml` (live `[langextract]`) but
  MUST NOT be `cfg/nemo_specs.toml` (KeyError, §8 R3 / §2).
- No network, file-handoff, or schema interface changes — config-only.

---

## 5. Configuration Schema (effective values after the edit)

| File | Key | Value |
|---|---|---|
| `cfg/nemo.toml` & `cfg/nemo_specs.toml` | top-level tables | exactly `chunking`, `embedding`, `general`, `langextract`, `llm` |
| `cfg/nemo.toml` | `[general].output_dir` | `./data/nemo_briefs_20260429` (+ inline history comment) — unchanged |
| `cfg/nemo.toml` | `[general].data_dir` | `./rawdata/parsed-techbriefs` (+ inline history comment) — unchanged |
| `cfg/nemo.toml` | `[llm].model` | `gpt-4o-mini` — unchanged |
| `cfg/nemo.toml` | `[chunking].relevance_concurrency` | `8` — unchanged |
| `cfg/nemo_specs.toml` | `[general].output_dir` | `./data/specs_20260516` — unchanged |
| `cfg/nemo_specs.toml` | `[general].data_dir` | `./rawdata-pubs/parsed-specs` — unchanged |
| `cfg/nemo_specs.toml` | `[llm].model` | `gpt-oss:120b#"gpt-oss:20b"#…` — unchanged |
| `cfg/nemo_specs.toml` | `[chunking].relevance_concurrency` | `2` — unchanged |
| both | `[llm]`, `[embedding]`, `[chunking]`, `[langextract]` (all keys) | byte-for-byte unchanged |
| both | header comment | one `#` line: consumed only by `_nemo.py` (+ `extract_artifacts.py` for `[langextract]` via `--cfg cfg/nemo.toml`) |

Removed entirely from both: `[general].metadata_folder`, `[general.theme]`,
`[general.tasks]`, `[pub242]`(+`.tasks`,`.saliency`), `[jsa]`(+`.tasks`),
`[nlp]`(+`.tasks`), `[doc]`(+`.rag`,`.mlp`,`.tasks`).

---

## 6. Data Flow / Artifacts

No on-disk pipeline artifact changes. The only files that change content are
`cfg/nemo.toml` and `cfg/nemo_specs.toml`. The §9 smoke writes throwaway chunk JSON under
`/tmp/cfgtest/doc-chunks_{size}_random_logical/` (scratch, not a tracked artifact, not the real
`output_dir`). No `*-logic-chunks.json`, `*-relevance.json`, `*-logic-ctx.json`,
`*-logic-artifacts.json`, `generated-questions.json`, or `self-check-qa-results.json` schema or
path changes.

---

## 7. Prerequisites / Assumptions

1. Interpreter: `.venv/bin/python`.
2. `cfg/nemo.toml` and `cfg/nemo_specs.toml` are at the audited 143-line layout (verify before
   editing; the §3 line ranges are exact for the current working tree). If either file has
   drifted, re-locate the DELETE blocks by table header, not by line number.
3. The §9 chunk smoke imports `_nemo.py`, which constructs an `Embedder` default and calls
   `ollama_api.list_models()` at import (CLAUDE.md). **Ollama must be reachable** at
   `http://localhost:11434` for the smoke step, even though `--chunk-only` never embeds;
   `nomic-embed-text:latest` should appear in `ollama list`. The import also builds OpenAI/
   Google registries (`aisa/gen/providers.py`) → **non-empty** `OPENAI_API_KEY` and
   `GOOGLE_API_KEY` in env/`.env` are required just to *import* `_nemo.py` (never used by a
   `--chunk-only` run; no network egress on the chunk path). The import-time default-arg
   `Embedder(EmbedConfig())` also constructs HF `all-MiniLM-L6-v2` — must be cached/reachable.
4. A small 1-doc corpus (`small_corpus/` or one copied `.md`) is available for the §9 smoke.
5. `git diff --stat` is the FR-5.1 gate; no other tracked file may change.

---

## 8. Risks & Mitigations

| # | Risk | Mitigation |
|---|---|---|
| R1 | A **future** code path (or a not-yet-audited sibling tool) starts reading a removed section, silently `KeyError`-ing or `.get()`-defaulting to wrong behavior | §2 grep proves zero current readers; the FR-3 header records the consumer contract so re-adders see it; §9 step-1 `read_toml` assertion pins the kept set; the deleted keys carry no production behavior today (all `tasks` flags were `false`; analysis sections feed no pipeline mode) |
| R2 | Editing the wrong block / clipping a kept section (e.g. trimming into `[chunking]` or dropping a `[llm]` key) → Pydantic `TypeError` at `_nemo.py:842-843` or a chunker `KeyError` | §9 step-1 asserts the exact kept-set; §9 step-2 (`--chunk-only`) exercises `LLMConfig`/`EmbedConfig` unpack + `get_chunker` and **fails loudly** on any clipped key; FR-2.3 forbids touching `[llm]`/`[embedding]` keys |
| R3 | Someone points `extract_artifacts.py --cfg cfg/nemo_specs.toml` after the cleanup and hits `KeyError` (`[langextract]` w/o `[artifact_extraction]`, `extract_artifacts.py:962-964`) — a pre-existing trap, not introduced here | Out of scope to fix; documented in §2 and the FR-3 header note; the specs route uses `extract_artifacts_specs.toml` per `srs-ollama-random-logical-pipeline.md` |
| R4 | The two files drift structurally during the edit (parity broken — FR-4) | Apply the *identical* structural deletion to both; §9 step-1 asserts the same kept set for both; FR-4.2 structural-diff check |
| R5 | Conservative scope leaves real redundancy (`logical_*` dead for the only route ever run, `[embedding]` fallback duplicates) — looks "not done" | Explicitly bounded by the user-chosen conservative scope; recorded as findings in §2 and deferred in §10 (moderate/aggressive trims), not silently dropped |
| R6 | `cfg/nemo_specs.toml` is git-untracked in some checkouts (it appears under `??` in status) → `git diff --stat` may not show it | Verify the file is staged/tracked before relying on the FR-5.1 diff gate; if untracked, assert its content directly and note it in the change description |

---

## 9. Acceptance Criteria / Test Plan

**Pre-flight (must hold before editing):**
- `wc -l cfg/nemo.toml cfg/nemo_specs.toml` == 143 each; the §3 DELETE blocks are at the cited
  headers. Ollama reachable; non-empty `OPENAI_API_KEY`/`GOOGLE_API_KEY` in env/`.env`.

**Step 1 — kept-set assertion (the core acceptance gate).**
```
.venv/bin/python -c "from aisa.utils import files; print(sorted(files.read_toml('cfg/nemo.toml')))"
.venv/bin/python -c "from aisa.utils import files; print(sorted(files.read_toml('cfg/nemo_specs.toml')))"
```
Both MUST print exactly `['chunking', 'embedding', 'general', 'langextract', 'llm']` (FR-2.1,
FR-4.1).

**Step 2 — 1-doc `--chunk-only` smoke into a scratch dir (idempotent stages → one doc is
sufficient).** With Ollama up and bogus-but-non-empty keys not required (real keys are inert on
the chunk path, but the import gate still needs them non-empty — §7.3):
```
mkdir -p /tmp/cfgtest-in && cp <one>.md /tmp/cfgtest-in/
.venv/bin/python _nemo.py --chunk-only --cfg cfg/nemo.toml \
    --input_dir /tmp/cfgtest-in --output_dir /tmp/cfgtest
```
MUST create `/tmp/cfgtest/doc-chunks_*_random_logical/` with at least the doc's `-chunks.json`
+ `-logic-chunks.json` and exit 0 — **no `KeyError`, no Pydantic `TypeError`/validation error**
(FR-5.2, R2). This exercises the `[general]` explicit-key reads, the `[llm]`/`[embedding]`
Pydantic unpack, and `get_chunker` over `[chunking]`. Repeating step 2 with
`--cfg cfg/nemo_specs.toml` (its `output_dir` already a real path — use a scratch
`--output_dir`) is an optional parity check.

**Step 3 — diff gate.** `git diff --stat` (and `git status` for the untracked-file caveat, R6)
MUST show only `cfg/nemo.toml`, `cfg/nemo_specs.toml`, and the two new `plans/*.md` changed —
**no `.py` file** (FR-5.1).

**Acceptance statement:** the feature is accepted when Step 1 prints the exact 5-key set for
both files, Step 2 completes with the scratch chunk dir and zero `KeyError`/Pydantic error, and
Step 3 confirms no source file changed.

**Full-run note (optional, recommended).** The cleanup is config-hygiene only; idempotent
file-handoff stages make the 1-doc chunk smoke sufficient for acceptance. To additionally
confirm the full Route B chain is unaffected, run the `pipeline-smoke-runner` agent over
`--sdg-logical → extract_artifacts.py → generate-qa.py → self-check-qa.py` (per
`srs-ollama-random-logical-pipeline.md §9`) on a 1-doc subset — expected: identical behavior to
pre-cleanup. This is a confidence check, not a gate.

---

## 10. Future Work (out of scope)

- **Moderate trim:** remove keys *inside* actively-unpacked sections that no mode functionally
  consumes — `[llm].chunk_size`, `[llm].max_chain_tokens`, `[embedding].prefix`, and (only if
  `method="logical"` is permanently retired) the `logical_*` keys; collapse the
  `[embedding].chunk_size`/`recursive_overlap` fallback duplicate. Requires per-key Pydantic-
  field analysis and a touch of `LLMConfig`/`EmbedConfig`, so deliberately deferred.
- **Aggressive trim:** strip both files down to only the `random_logical` route's live keys, or
  merge `cfg/nemo.toml`/`cfg/nemo_specs.toml` into one config with a profile switch. Deferred.
- Remove the dead `[langextract]` copy from `cfg/nemo_specs.toml` once the specs route's
  step-2 config (`extract_artifacts_specs.toml`) is the documented invariant and parity is no
  longer desired.
- A CI/`read_toml` guard test asserting the FR-2.1 kept-set so carryover cruft cannot regress.
