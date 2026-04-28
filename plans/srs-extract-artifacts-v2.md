# Software Requirements Specification: Extract-Artifacts v2 — Normative 21-Class Taxonomy

**Feature:** Replace the 8-class pavement-noun extraction taxonomy in the standalone `extract_artifacts.py` with a 21-class normative/functional taxonomy backed by a new prompt and a new few-shot example. Plumbing (CLI, schema, idempotency, provider) is preserved.
**Component:** `nvidia-sdq_custom`
**Version:** 0.2 (draft)
**Status:** Proposed
**Companion plan:** `plans/plan-extract-artifacts-v2.md`

---

## 1. Introduction

### 1.1 Purpose
This SRS defines requirements for swapping the artifact taxonomy used by `extract_artifacts.py` from a domain-noun-focused vocabulary (current v1: `material`, `distress`, `treatment`, `specification`, `test_method`, `metric`, `process`, `reference`) to a normative/functional vocabulary (v2: 21 classes covering rules, conditions, methods, parameters, findings, recommendations, evidence, etc.) better suited to downstream QA generation.

### 1.2 Scope

In scope:
- New prompt file `prompts/nemo_logic-artifacts-02.txt`.
- Replacement of the `EXTRACTION_CLASSES` list and `PAVEMENT_EXAMPLES` few-shot data in `extract_artifacts.py`.
- Two changed `LXConfig` defaults (`extraction_passes`, `prompt_name`). No new fields; `lx.extract` call signature unchanged from v1.
- **Per-extraction schema change**: top-level keys become `{artifact_id, text, description, significance, char_interval, attributes}`. `description` and `significance` are removed from `attributes` and elevated to top-level. `description` is always present (default `""`); `significance` is `null` when not stated by the source.
- Updates to `extract_artifacts.toml` `[langextract]` defaults.
- Updates to `docs/qa-generation.md` Step 2 section.

Out of scope:
- Provider swap. v1's OpenAI `gpt-4o-mini` and `OPENAI_API_KEY` resolution stay (user-confirmed).
- Doc-level wrapper (`{doc_id, artifacts: [{chunk_id, tokens, extractions, [error]}]}`) — unchanged from v1.
- `artifact_id` format (`{doc_id}_chunk_{chunk_id}_art_{idx}`) — unchanged from v1.
- Mode-3 guard, idempotency semantics, failure-isolation behavior.
- `lx.extract` call signature (no new kwargs; `max_char_buffer` stays at 10000; no `max_workers`). Sub-chunking is not desired here — mode-3 chunks are pre-sized to be coherent and fit in one LLM call.
- Few-shot example **input format** — `lx.data.Extraction.attributes` continues to carry `description` and `significance` (langextract requires them there). Only the script's **output transformation** moves them to top-level keys.
- Type-specific attribute enforcement in code (prompt + example are the only steering mechanism).
- Stages 2–4 of the broader pipeline (artifact filter, QA generation, QA validation).
- Deletion of the v1 prompt file `prompts/nemo_logic-artifacts.txt` (kept on disk for reference; deletion deferred).
- Bumping few-shot coverage beyond the single example provided in §10 of the recommendation (5 of 21 classes covered).
- Migration tooling for existing v1 `-logic-artifacts.json` files (operator regenerates with `--overwrite`).

### 1.3 Definitions
- **v1** — current extract_artifacts state, 8-class pavement-noun taxonomy.
- **v2** — state introduced by this SRS, 21-class normative/functional taxonomy.
- **Normative class** — extraction class describing what the text *does* (e.g. imposes a requirement, states a condition, reports a finding) rather than the domain entity it names.
- **Do-not-extract class** — class name explicitly forbidden by the v2 prompt: `table`, `figure`, `reference`, `metadata`, `section_title`, `page_number`, `header`, `footer`, `table_of_contents_entry`, `caption_alone`, `revision_date`, `document_title`. Mentions of these inside meaningful artifacts are preserved as `attributes.context_reference`.

### 1.4 References
- `plans/plan-extract-artifacts-v2.md` — companion implementation plan.
- `plans/plan-sdg-logical-step2.md` and `plans/srs-sdg-logical-step2.md` — original Step 2 plan and SRS (v1).
- The 21-class taxonomy specification provided by the user (§§1–13 of the recommendation message).

---

## 2. Overall Description

### 2.1 Product Perspective
The v1 `extract_artifacts.py` produces artifacts grouped by 8 domain-noun classes — useful for entity-style retrieval but undersuited to QA generation, which needs to ask about rules, conditions, methods, and decisions. v2 keeps every line of plumbing (file walk, idempotency, mode-3 guard, provider, doc-level wrapper) and replaces the vocabulary, prompt, example, two `LXConfig` defaults, and the per-extraction shape. The doc-level wrapper is byte-compatible with v1; the per-extraction shape is a deliberate v1→v2 break (described in §3 FR-6.1, §3 FR-6.7, and §4 NFR-1).

### 2.2 User Classes
- **Pipeline operator** — runs `python extract_artifacts.py` after Step 1 to produce artifact files for the corpus. Sees richer, normatively-typed artifacts in v2.
- **Pipeline developer** — extends or tunes the taxonomy and few-shot examples. Reads the prompt file directly.

### 2.3 Operating Environment
Identical to v1: Python 3.11+ (venv at 3.14), `langextract==1.2.1`, `loguru`, `python-dotenv`, `OPENAI_API_KEY` populated in `.env` or environment.

### 2.4 Constraints
- The 21-class vocabulary SHALL be defined in code as a frozen list. No runtime mutation.
- The do-not-extract class names SHALL be enforced by the same code path that drops out-of-vocab classes — i.e. the existing class-vocab gate in `PavementExtractor.extract`.
- The doc-level shape (`{doc_id, artifacts: [...]}`) and the per-chunk wrapper (`{chunk_id, tokens, extractions, [error]}`) SHALL NOT change.
- The per-extraction shape SHALL change: `description` and `significance` are promoted from `attributes` to top-level keys. New per-extraction key set: `{artifact_id, text, description, significance, char_interval, attributes}`. `attributes` SHALL contain only type-specific keys and the remaining common attributes (`subject`, `scope`, `context_reference`, `source_cue`).
- The provider SHALL remain OpenAI `gpt-4o-mini` with `OPENAI_API_KEY` resolution unchanged.
- `lx.extract` call signature SHALL be unchanged from v1 (`max_char_buffer` stays at 10000; no `max_workers` kwarg).
- The new prompt file SHALL be loaded the same way as v1 — via `Path(cfg.prompt_lib) / f"{cfg.prompt_name}.txt"`.
- Type-specific attribute schemas (`requirement.modality`, `parameter.symbol`, etc.) SHALL NOT be enforced by the script. The script validates only the class name.

### 2.5 Assumptions
- `prompts/nemo_logic-artifacts-02.txt` will be created with the exact `prompt_description` text from §9 of the recommendation.
- `PAVEMENT_EXAMPLES` will be replaced with the exact 6-extraction example from §10 of the recommendation.
- The operator regenerates stale v1 outputs via `--overwrite` if v2 outputs are needed in a directory that already has v1 files.

---

## 3. Functional Requirements

### FR-1 Extraction-class vocabulary (v2)
**FR-1.1** `extract_artifacts.py::EXTRACTION_CLASSES` SHALL be exactly:
```python
[
    "requirement", "condition", "exception", "constraint",
    "procedure", "method", "formula", "parameter",
    "threshold", "definition", "actor_role", "deliverable",
    "assumption", "finding", "recommendation", "best_practice",
    "decision", "rationale", "issue", "risk", "evidence",
]
```
(21 elements, in that order.)

**FR-1.2** Any extraction whose `extraction_class` is not in `EXTRACTION_CLASSES` SHALL be dropped, and the script SHALL emit one `"NLP"`-level log line per drop naming the doc_id, chunk_id, and offending class.

**FR-1.3** The prompt SHALL explicitly forbid the do-not-extract class names. If the model nonetheless emits any of them, FR-1.2 catches them at the same gate.

### FR-2 New prompt file
**FR-2.1** The prompt SHALL live at `prompts/nemo_logic-artifacts-02.txt`.
**FR-2.2** Its content SHALL be the verbatim text of the `prompt_description` block in §9 of the recommendation, including: the extraction-task description, the 21-class definition list, the common-attribute glossary, the boundary rules, and the do-not-extract list.
**FR-2.3** The file SHALL contain no `{placeholders}` (langextract supplies the input via `text_or_documents`).
**FR-2.4** The v1 prompt file `prompts/nemo_logic-artifacts.txt` SHALL be left untouched on disk. Deletion is deferred (out of scope here).

### FR-3 Few-shot examples
**FR-3.1** `extract_artifacts.py::PAVEMENT_EXAMPLES` SHALL contain exactly one `lx.data.ExampleData`.
**FR-3.2** That example's `text` SHALL be the verbatim ESALs/design-analyses/drainage paragraph from §10 of the recommendation.
**FR-3.3** That example SHALL contain exactly six `lx.data.Extraction` objects with class names, `extraction_text` spans, and `attributes` dicts matching §10 verbatim. The classes covered are: `condition`, `requirement` (×2), `actor_role`, `deliverable`, `best_practice`.
**FR-3.4** Source spans (`extraction_text`) SHALL be verbatim substrings of the example `text` (no paraphrase, no whitespace normalization).

### FR-4 `LXConfig` field changes
**FR-4.1** `LXConfig.extraction_passes` default SHALL change from `3` to `2`.
**FR-4.2** `LXConfig.prompt_name` default SHALL change from `"nemo_logic-artifacts"` to `"nemo_logic-artifacts-02"`.
**FR-4.3** Other `LXConfig` fields (`model`, `api_key`, `temperature`, `max_char_buffer`, `prompt_lib`) SHALL retain their v1 defaults. No new fields are added.

### FR-5 `lx.extract` call
**FR-5.1** `PavementExtractor.extract` SHALL call `lx.extract` with the same kwargs as v1: `text_or_documents`, `prompt_description`, `examples`, `model_id`, `api_key`, `temperature`, `extraction_passes`, `max_char_buffer`, `show_progress=False`. No new kwargs.

### FR-6 Schema and transformation
**FR-6.1** Doc-level output shape SHALL be `{"doc_id": str, "artifacts": list}` — unchanged from v1. Per-chunk wrapper SHALL be `{chunk_id, tokens, extractions, [error]}` — unchanged from v1.
**FR-6.2** `artifact_id` format SHALL remain `f"{doc_id}_chunk_{chunk_id}_art_{ext_idx}"` with the same 0-based-across-classes counter.
**FR-6.3** Idempotency: if `{doc_id}-logic-artifacts.json` exists and `--overwrite` is not set, the doc SHALL be skipped with a `"CHUNK"`-level cache-hit log line. Identical to v1.
**FR-6.4** Failure isolation: a `langextract` exception for one chunk SHALL log at `"NLP"`, emit `"extractions": {}` + `"error": "<message>"` for that chunk, and continue. Identical to v1.
**FR-6.5** The mode-3 guard at `_nemo.py:439` and inside `main(cfg, overwrite)` / the `__main__` block SHALL remain in place unchanged.
**FR-6.6** Provider: `model="gpt-4o-mini"`, `api_key=os.getenv("OPENAI_API_KEY")` resolution. Unchanged.
**FR-6.7** **Per-extraction shape**: each entry inside `extractions[<class>]` SHALL have keys `{artifact_id, text, description, significance, char_interval, attributes}`. `description` SHALL be a string (default `""` if the model omits it). `significance` SHALL be either `null` or a non-empty string (default `null` if the model omits it; the v2 prompt instructs the model to populate it only when the source supports significance). `attributes` SHALL contain only type-specific keys (e.g. `modality`, `symbol`, `purpose`) and the remaining common attributes (`subject`, `scope`, `context_reference`, `source_cue`); `attributes` SHALL NOT contain `description` or `significance`.
**FR-6.8** **Transformation logic**: `PavementExtractor.extract` SHALL produce each output entry by (a) copying `ext.attributes` to a mutable dict, (b) popping `description` (default `""`) and `significance` (default `None`) from that dict, and (c) emitting the entry with `description` and `significance` as top-level keys and the popped dict as the `attributes` value. The transformation SHALL be the only place where the elevation happens — no other code path SHALL re-add `description` or `significance` to `attributes`.

### FR-7 `extract_artifacts.toml`
**FR-7.1** The `[langextract]` block SHALL set:
- `model = "gpt-4o-mini"` (unchanged)
- `temperature = 0.0` (unchanged)
- `extraction_passes = 2` (was 3)
- `max_char_buffer = 10000` (unchanged)
- `prompt_name = "nemo_logic-artifacts-02"` (was `"nemo_logic-artifacts"`)
- `prompt_lib = "./prompts"` (unchanged)
**FR-7.2** The `[paths]` section (`input_dir`) SHALL remain unchanged.
**FR-7.3** Comments in the file SHALL be updated where they reference the old defaults; the file SHALL still be loadable as TOML.

### FR-8 Documentation
**FR-8.1** `docs/qa-generation.md`'s Step 2 section SHALL replace the 8-row class table with a 21-row table — one row per class with a one-line definition mirroring §3 of the recommendation.
**FR-8.2** The schema JSON example in that section SHALL use one of the new classes (e.g. `requirement` or `parameter`) instead of `material`.
**FR-8.3** The configuration code block SHALL reflect the FR-7.1 defaults.
**FR-8.4** A one-line note SHALL state that do-not-extract types (`table`, `figure`, `reference`, etc.) appear only inside `attributes.context_reference` per the new prompt, never as standalone extractions.

### FR-9 Non-interference and file preservation
**FR-9.1** The v1 prompt file `prompts/nemo_logic-artifacts.txt` SHALL remain unmodified on disk.
**FR-9.2** No other module or script SHALL be edited by this feature beyond the four files enumerated above (`extract_artifacts.py`, `extract_artifacts.toml`, `prompts/nemo_logic-artifacts-02.txt` [new], `docs/qa-generation.md`).
**FR-9.3** Existing v1 `-logic-artifacts.json` files SHALL continue to be valid JSON; the script SHALL not corrupt them. Operators force-regenerate via `--overwrite`.

---

## 4. Non-Functional Requirements

### NFR-1 Backward compatibility (file-level)
v1 `-logic-artifacts.json` files SHALL remain valid JSON; the script's idempotency gate (FR-6.3) skip-writes them. They are NOT structurally compatible with v2 outputs: the `extractions.<class>` keys differ (8-class v1 → 21-class v2), and the per-extraction shape differs (v1: `{artifact_id, text, char_interval, attributes}` with `description`/`significance` inside `attributes`; v2: `{artifact_id, text, description, significance, char_interval, attributes}`). Consumers reading both versions MUST dispatch on schema. Operators regenerate v1 caches via `--overwrite`.

### NFR-2 Determinism
With `temperature=0.0`, repeated runs over identical input SHOULD produce identical output, modulo `langextract`'s internal retry/sampling behavior across `extraction_passes`. The script's bookkeeping (file walk order, per-doc loop, `artifact_id` indexing) remains deterministic.

### NFR-3 Observability
Every per-doc run SHALL emit a `"CHUNK"`-level summary or cache-hit line. Class-vocab drops SHALL emit one `"NLP"` line per drop (identical to v1, but now applies to a 21-class set). Failed chunks SHALL emit one `"NLP"` line per failure. Cost telemetry remains deferred (see OQ-3).

### NFR-4 Performance envelope
- `extraction_passes` reduced 3 → 2 ≈ 33 % fewer model calls per chunk vs. v1.
- `max_char_buffer` stays at 10000, so logical chunks (max ~8000 chars at `hybrid_window=8 × chunk_size=256` tokens) are not sub-chunked; one LLM call per logical chunk per pass. No change in call count from v1 on this axis.
- Net cost per chunk vs. v1: ~33 % cheaper.

### NFR-5 Schema invariant
The doc-level shape (`{doc_id, artifacts: [...]}`) and per-chunk wrapper SHALL be strict invariants v1 → v2. The per-extraction shape changes per FR-6.7: top-level keys are `{artifact_id, text, description, significance, char_interval, attributes}`. Schema-aware diffs between a v1 and v2 file for the same source chunk SHALL differ in: (a) `extractions.<class>` keys (different class vocabulary), (b) per-extraction key set (added `description`/`significance` at top, removed from `attributes`), (c) `attributes` content (now type-specific + remaining common attrs only).

### NFR-6 Dependency isolation
`extract_artifacts.py` SHALL remain free of `aisa.*` imports (NFR-6 from the v1 SRS carries forward unchanged).

---

## 5. Interfaces

### 5.1 CLI interface
```text
python extract_artifacts.py [--cfg PATH] [--input_dir DIR] [--overwrite]
```
**Unchanged from v1.**

### 5.2 Python interface
```python
EXTRACTION_CLASSES: list[str]                       # 21-class frozen vocabulary (FR-1.1)
PAVEMENT_EXAMPLES: list[lx.data.ExampleData]        # 1 example, 6 extractions (FR-3)

@dataclass
class LXConfig:
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    temperature: float = 0.0
    extraction_passes: int = 2                      # was 3
    max_char_buffer: int = 10000                    # unchanged
    prompt_name: str = "nemo_logic-artifacts-02"    # was "nemo_logic-artifacts"
    prompt_lib: str = "./prompts"

class PavementExtractor:
    def __init__(self, cfg: LXConfig) -> None: ...                # surface unchanged
    def extract(self, text: str, doc_id: str, chunk_id: int) -> dict[str, list[dict]]: ...

def main(cfg: dict, overwrite: bool = False) -> None: ...         # unchanged
```

### 5.3 File interface
Input: `{input_dir}/{doc_id}-logic-chunks.json` (unchanged from v1). Output: `{input_dir}/{doc_id}-logic-artifacts.json`. Schema:

```
{
  doc_id,
  artifacts: [
    {
      chunk_id, tokens,
      extractions: {
        <class>: [
          {artifact_id, text, description, significance, char_interval, attributes}
        ]
      },
      [error]
    }
  ]
}
```

Per-extraction notes: `description` is always present (string, possibly `""`); `significance` is `null` when not stated by the source; `attributes` contains only type-specific keys (`modality`, `symbol`, `purpose`, etc.) and the remaining common attrs (`subject`, `scope`, `context_reference`, `source_cue`).

### 5.4 Configuration interface
The `[langextract]` block in `extract_artifacts.toml` and (when applicable) `cfg/nemo.toml` SHALL match FR-7.1. The `[paths]` block (or `[general]` fallback) is unchanged.

### 5.5 Environment interface
- `OPENAI_API_KEY` — required. Resolution unchanged from v1.

### 5.6 Prompt interface
File: `prompts/nemo_logic-artifacts-02.txt`. Plain text. Verbatim from §9 of the recommendation. No `{placeholders}`.

---

## 6. Acceptance Criteria

- **AC-1** `prompts/nemo_logic-artifacts-02.txt` exists and is byte-equal to the §9 prompt block.
- **AC-2** `python -c "from extract_artifacts import EXTRACTION_CLASSES; print(EXTRACTION_CLASSES)"` outputs the 21-class list in FR-1.1 order.
- **AC-3** `python -c "from extract_artifacts import PAVEMENT_EXAMPLES; print(len(PAVEMENT_EXAMPLES), len(PAVEMENT_EXAMPLES[0].extractions))"` outputs `1 6`.
- **AC-4** `LXConfig()` instance has `extraction_passes==2` and `prompt_name=="nemo_logic-artifacts-02"`. Other defaults (`max_char_buffer==10000`, etc.) unchanged.
- **AC-5** Cold run on the existing TBF000027 fixture writes `-logic-artifacts.json` with the v2 schema: top-level `{doc_id, artifacts}`; per-chunk `{chunk_id, tokens, extractions, [error]}`; per-extraction keys are exactly `{artifact_id, text, description, significance, char_interval, attributes}`.
- **AC-6** Across the cold-run output, every emitted class name is in the 21-class set (no `material`/`distress`/etc.; no `table`/`figure`/`reference`/etc.). Verified by aggregator regex.
- **AC-7** At least one extraction has class `requirement` and `attributes.modality ∈ {"shall", "must", "required", "prohibited", "may not", "is to be"}`. (Sanity-check the prompt is producing modality information.)
- **AC-8** All `artifact_id`s match `^.+_chunk_\d+_art_\d+$`; chunk_id round-trips; uniqueness within the output dir holds. Identical to v1 AC-4.
- **AC-9** Idempotency: second invocation with no `--overwrite` writes nothing (mtime unchanged) and makes zero `langextract` calls. Identical to v1 AC-8.
- **AC-10** Mode-3 guard (`extract_artifacts.py` and `_nemo.py:439`) still rejects `recursive` and `logical` with the redirect error. Identical to v1 AC-6/AC-7.
- **AC-11** `grep -E '^\s*(extraction_passes|prompt_name)\s*=' extract_artifacts.toml` outputs the two FR-7.1 lines with the v2 values.
- **AC-12** `docs/qa-generation.md` Step 2 section's class table contains 21 rows; the do-not-extract note is present; the schema JSON example uses a v2 class name and the new top-level `description`/`significance` keys.
- **AC-13** A pre-existing v1 `-logic-artifacts.json` file is left in place on a re-run without `--overwrite` (cache hit). With `--overwrite`, it is regenerated under v2.
- **AC-14** Non-interference: running `--sdg` for any chunking method continues to work and writes its bundled-flow files unchanged. Same for `--sdg-logical`.
- **AC-15** Per-extraction shape verification: every emitted entry has `description` as a string (possibly `""`); `significance` is `null` or a non-empty string; `attributes` does NOT contain a `description` or `significance` key. Verified by aggregator across all extractions in the cold-run output.

---

## 7. Risks and Open Questions

### 7.1 Risks

- **R-1** **Class drift.** With 21 candidate classes and overlapping boundaries (e.g. `requirement` vs `recommendation` vs `best_practice`, `method` vs `procedure`), the model may consistently pick a sibling class for an artifact that fits another. Mitigation: the boundary-rules block in the v2 prompt is explicit on the most common confusions; FR-1.2 logs let us measure drift over a real corpus.
- **R-2** **Few-shot under-coverage.** The single 6-extraction example covers 5 of 21 classes. Classes never demonstrated in the example set (`exception`, `constraint`, `procedure`, `method`, `formula`, `parameter`, `threshold`, `definition`, `assumption`, `finding`, `recommendation`, `decision`, `rationale`, `issue`, `risk`, `evidence`) rely entirely on the prompt's class definitions for steering. Mitigation: monitor class-distribution histograms in early runs; supplement with manual- and report-style examples once gaps are visible.
- **R-3** **Mixed v1/v2 outputs in the same directory.** A directory that already holds v1 `-logic-artifacts.json` files will skip-write under v2 (FR-6.3 idempotency). The two file shapes are not interchangeable for downstream consumers (NFR-1). Mitigation: documented in `docs/qa-generation.md`; visible from the cache-hit log line; force regeneration via `--overwrite`.
- **R-4** **Type-specific attribute drift.** The script does not enforce per-class attribute schemas (e.g. requiring `modality` on `requirement`, `symbol` on `parameter`). The prompt is the only steer. Some classes may emit incomplete or extra attributes. Mitigation: enforcement is a candidate follow-up if drift becomes painful (see OQ-4).
- **R-5** **`description` omission by the model.** The v2 prompt instructs the model to provide a normalized `description` for every extraction, but the model may omit it. Mitigation: FR-6.7 defaults `description` to `""` so the key is always present; missing-description rate is observable via post-hoc inspection. If empty descriptions become frequent, tighten the prompt.

### 7.2 Open questions (non-blocking)

- **OQ-1** **Delete the v1 prompt file?** Once v2 is validated on a real corpus, `prompts/nemo_logic-artifacts.txt` becomes dead code. Defer to a cleanup PR.
- **OQ-2** **Add report-style few-shot.** Section §13 of the recommendation lists `finding`/`recommendation`/`decision`/`issue`/`risk`/`evidence`/`rationale` as report-dominant classes. The corpus is mostly tech briefs (research-style), so these matter. A second `ExampleData` covering them is a likely follow-up.
- **OQ-3** **Cost telemetry.** Carries forward from v1 SRS OQ-2. Still deferred.
- **OQ-4** **Per-class attribute validator.** If R-5 manifests, a post-hoc validator matching `requirement → modality`, `parameter → symbol`, etc., from §5 of the recommendation can run after extraction and either drop offenders or flag them. Defer.
- **OQ-5** **Schema versioning marker.** Should v2 outputs carry an explicit `"schema_version": 2` (or similar) at the top level so consumers can disambiguate v1/v2 caches in the same directory? With v2's per-extraction shape change (top-level `description`/`significance`), per-extraction key sets disambiguate too — but a top-level marker is more robust for downstream filter/QA-gen tools. Defer until a consumer needs it.
