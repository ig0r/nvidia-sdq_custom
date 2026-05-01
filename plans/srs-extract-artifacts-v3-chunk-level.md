# Software Requirements Specification: Extract-Artifacts v3 — Chunk-Level Artifacts

**Feature:** Add three chunk-level artifact classes (`chunk_summary`, `chunk_topic`, `pavement_engineering_term`) to the standalone `extract_artifacts.py` pipeline, on top of the v2 21-class span-level taxonomy. Chunk-level artifacts are produced by a *second* `lx.extract` call per chunk against a dedicated prompt and few-shot example set. Span-level extraction behavior is preserved verbatim from v2.
**Component:** `nvidia-sdq_custom`
**Version:** 0.3 (draft)
**Status:** Proposed
**Companion plan:** `plans/plan-extract-artifacts-v3-chunk-level.md`

---

## 1. Introduction

### 1.1 Purpose
This SRS defines requirements for extending `extract_artifacts.py` from a single-call, span-only extraction into a two-call, span+chunk extraction. The goal is to produce compact chunk-wide signals (a one-sentence summary, 1-5 normalized topics, 0+ canonical pavement-engineering terms) alongside the existing 21-class span-level artifacts. Span-level behavior — taxonomy, prompt body, example set, attribute schemas, per-extraction shape — is unchanged from v2.

### 1.2 Scope

In scope:
- New prompt files `prompts/nemo_logic-artifacts-03-span.txt` (byte-equal copy of v2 prompt) and `prompts/nemo_logic-artifacts-03-chunk.txt` (new chunk-level prompt).
- Extension of `EXTRACTION_CLASSES` from 21 → 24, with two named subsets `SPAN_LEVEL_CLASSES` and `CHUNK_LEVEL_CLASSES`.
- Rename of v2 `PAVEMENT_EXAMPLES` to `SPAN_LEVEL_EXAMPLES` (content unchanged). Addition of `CHUNK_LEVEL_EXAMPLES`.
- `LXConfig` field changes: gain `prompt_name_span`, `prompt_name_chunk`, `chunk_extraction_passes`; remove `prompt_name`. `extraction_passes` retained, scoped to span-level.
- `PavementExtractor.extract` becomes a two-call orchestrator (span + chunk), with per-call class gates and per-class-group output transformation.
- **Per-extraction shape by class group**: span-level entries use the v2 6-key shape (`{artifact_id, text, description, significance, char_interval, attributes}`); chunk-level entries use a 4-key shape (`{artifact_id, text, char_interval, attributes}`).
- Updates to `extract_artifacts.toml` `[langextract]` for the new field names.
- Updates to `docs/qa-generation.md` Step 2 section.

Out of scope:
- Provider swap. v2's OpenAI `gpt-4o-mini` and `OPENAI_API_KEY` resolution stay (user-confirmed across v1/v2/v3).
- Doc-level wrapper (`{doc_id, artifacts: [{chunk_id, tokens, extractions, [error]}]}`) — unchanged from v2.
- `artifact_id` format (`{doc_id}_chunk_{chunk_id}_art_{idx}`) — unchanged from v2; counter continuous across both calls.
- Mode-3 guard, idempotency semantics, failure-isolation behavior — unchanged.
- `lx.extract` call signature for the span-level call — unchanged from v2 (`max_char_buffer=10000`, no `max_workers`).
- Span-level prompt body — `prompts/nemo_logic-artifacts-03-span.txt` is byte-equal to `prompts/nemo_logic-artifacts-02.txt`.
- Span-level few-shot content — `SPAN_LEVEL_EXAMPLES` is byte-equal to v2 `PAVEMENT_EXAMPLES`.
- Code-side enforcement of chunk-level quantity rules (exactly 1 `chunk_summary`, 1-5 `chunk_topic`).
- Code-side validation of topic / term-category vocabularies.
- Parallel `lx.extract` calls (asyncio / threading).
- Migration tooling for v2 outputs.
- Deletion of v1 / v2 prompt files.
- Schema-version marker at the doc level (v2 SRS OQ-5).

### 1.3 Definitions
- **v2** — prior state, 21 span-level classes, single `lx.extract` call per chunk.
- **v3** — state introduced by this SRS, 24 classes total, two `lx.extract` calls per chunk (span + chunk).
- **Span-level artifact** — extraction class describing a specific source clause (the v2 21 classes).
- **Chunk-level artifact** — extraction class describing a property of the whole chunk: `chunk_summary`, `chunk_topic`, `pavement_engineering_term`.
- **Chunk-summary span** — verbatim last complete sentence of the chunk used as `extraction_text` for `chunk_summary` entries (or the closest preceding complete sentence if the chunk ends mid-fragment).
- **Class group** — `SPAN_LEVEL_CLASSES` or `CHUNK_LEVEL_CLASSES`. Per-extraction output shape depends on which group a class belongs to.

### 1.4 References
- `plans/plan-extract-artifacts-v3-chunk-level.md` — companion implementation plan.
- `plans/plan-extract-artifacts-v2.md`, `plans/srs-extract-artifacts-v2.md` — previous plan and SRS (v2).
- The chunk-level taxonomy specification provided by the user (CHUNK_LEVEL_ARTIFACT_TYPES, CHUNK_LEVEL_TYPE_SPECIFIC_ATTRIBUTES, PAVEMENT_TOPIC_EXAMPLES, PAVEMENT_TERM_CATEGORIES, recommended `chunk_level_prompt_block`).

---

## 2. Overall Description

### 2.1 Product Perspective
v2 produces normatively-typed span-level artifacts but exposes no chunk-wide signals. Downstream retrieval / filtering / QA generation needs a one-sentence chunk overview, a small set of normalized topic labels, and a typed canonical-term inventory. v3 keeps every line of v2 plumbing (file walk, idempotency, mode-3 guard, provider, doc-level wrapper, per-chunk wrapper) and adds:
- A second `lx.extract` call per chunk with a dedicated prompt and example set.
- Three new classes that surface in the same `extractions` dict as the existing 21.
- A class-group-aware output transformation (span-level entries keep v2 shape; chunk-level entries use a compact 4-key shape).

The doc-level wrapper is byte-compatible v2→v3. The per-chunk wrapper is unchanged. The per-extraction shape varies by class group and is therefore a deliberate v2→v3 break (described in §3 FR-6 and §4 NFR-1).

### 2.2 User Classes
- **Pipeline operator** — runs `python extract_artifacts.py` after Step 1 to produce artifact files.
- **Pipeline developer** — extends or tunes the taxonomy, prompts, and few-shot examples.
- **Downstream consumer** — reads chunk-level signals (`chunk_summary.summary`, `chunk_topic.topic`, `pavement_engineering_term.normalized_term`) for retrieval, filtering, QA generation, and entity linking.

### 2.3 Operating Environment
Identical to v2: Python 3.11+ (venv at 3.12), `langextract==1.2.1`, `loguru`, `python-dotenv`, `OPENAI_API_KEY` populated in `.env` or environment.

### 2.4 Constraints
- The 24-class vocabulary SHALL be defined in code as the union of two frozen lists `SPAN_LEVEL_CLASSES` (21 elements) and `CHUNK_LEVEL_CLASSES` (3 elements). The two SHALL be disjoint. No runtime mutation.
- The doc-level shape (`{doc_id, artifacts: [...]}`) and the per-chunk wrapper (`{chunk_id, tokens, extractions, [error]}`) SHALL NOT change.
- The per-extraction shape SHALL depend on class group:
  - Span-level: `{artifact_id, text, description, significance, char_interval, attributes}` (v2 shape).
  - Chunk-level: `{artifact_id, text, char_interval, attributes}` — `description` and `significance` SHALL NOT appear at top level, and SHALL NOT be popped from chunk-level `attributes` if the model emits them (they remain in `attributes` as residual keys, which the prompt does not request).
- The provider SHALL remain OpenAI `gpt-4o-mini` with `OPENAI_API_KEY` resolution unchanged.
- `lx.extract` call signature for both calls SHALL be the v2 shape: `text_or_documents`, `prompt_description`, `examples`, `model_id`, `api_key`, `temperature`, `extraction_passes`, `max_char_buffer=10000`, `show_progress=False`. No `max_workers`.
- Both prompt files SHALL be loaded the same way as v2 — via `Path(cfg.prompt_lib) / f"{cfg.prompt_name_X}.txt"`.
- Quantity rules for chunk-level types (`exactly 1 chunk_summary`, `1-5 chunk_topic`, `0+ pavement_engineering_term`) SHALL be expressed in the prompt only; the script SHALL NOT enforce them in code.
- Topic and term-category vocabularies SHALL be enumerated in the prompt only; the script SHALL NOT validate them.

### 2.5 Assumptions
- `prompts/nemo_logic-artifacts-03-span.txt` will be created as a byte-equal copy of `prompts/nemo_logic-artifacts-02.txt`.
- `prompts/nemo_logic-artifacts-03-chunk.txt` will be created with the chunk-level prompt body (3 classes, attribute schemas, quantity rules, span convention, topic vocab, term-category vocab, strict source-grounding rule).
- The operator regenerates stale v2 outputs via `--overwrite` if v3 outputs are needed in a directory that already has v2 files.
- The model emits class names that are valid Python identifiers and are members of one of the two class groups; cross-mode emissions (span call returning a chunk class or vice versa) are handled by per-call class gates.

---

## 3. Functional Requirements

### FR-1 Extraction-class vocabulary (v3)
**FR-1.1** `extract_artifacts.py::SPAN_LEVEL_CLASSES` SHALL be exactly the v2 21-class list:
```python
[
    "requirement", "condition", "exception", "constraint",
    "procedure", "method", "formula", "parameter",
    "threshold", "definition", "actor_role", "deliverable",
    "assumption", "finding", "recommendation", "best_practice",
    "decision", "rationale", "issue", "risk", "evidence",
]
```
in that order.

**FR-1.2** `extract_artifacts.py::CHUNK_LEVEL_CLASSES` SHALL be exactly:
```python
["chunk_summary", "chunk_topic", "pavement_engineering_term"]
```
in that order.

**FR-1.3** `EXTRACTION_CLASSES` SHALL equal `SPAN_LEVEL_CLASSES + CHUNK_LEVEL_CLASSES` (24 elements). The two subsets SHALL be disjoint.

**FR-1.4** Any extraction returned by the **span-level** call whose class is not in `SPAN_LEVEL_CLASSES` SHALL be dropped, and the script SHALL emit one `"NLP"`-level log line naming the doc_id, chunk_id, the offending class, and "span call".

**FR-1.5** Any extraction returned by the **chunk-level** call whose class is not in `CHUNK_LEVEL_CLASSES` SHALL be dropped, and the script SHALL emit one `"NLP"`-level log line naming the doc_id, chunk_id, the offending class, and "chunk call".

### FR-2 Prompt files
**FR-2.1** A new prompt SHALL exist at `prompts/nemo_logic-artifacts-03-span.txt`. Its contents SHALL be byte-equal to `prompts/nemo_logic-artifacts-02.txt`.

**FR-2.2** A new prompt SHALL exist at `prompts/nemo_logic-artifacts-03-chunk.txt`. It SHALL contain:
- The 3 chunk-level class definitions.
- Quantity rules: exactly 1 `chunk_summary`, 1-5 `chunk_topic`, 0+ `pavement_engineering_term`.
- The `chunk_summary` span convention: *"Set extraction_text to the last complete sentence of the chunk. If the chunk ends mid-fragment, use the closest preceding complete sentence."*
- Per-class attribute schemas:
  - `chunk_summary`: `summary`, `document_function`, `scope`.
  - `chunk_topic`: `topic`, `topic_role`.
  - `pavement_engineering_term`: `term`, `normalized_term`, `term_category`.
- The recommended topic vocabulary (rigid pavement design, flexible pavement design, traffic loading, drainage, materials, construction, maintenance, rehabilitation, approval workflow, …).
- The term-category vocabulary (traffic, pavement_type, design_parameter, material, layer, method, test_method, distress, construction, maintenance, organization, form, software, other).
- The strict source-grounding rule for terms (no inferred or related terms; no generic words unless part of a meaningful technical term).

**FR-2.3** Both prompt files SHALL contain no `{placeholders}`.

**FR-2.4** The chunk-level prompt SHALL NOT mention `description` or `significance` (those keys are span-level only; the chunk-level prompt stays silent on them so the model does not emit them).

**FR-2.5** The v1 prompt file `prompts/nemo_logic-artifacts.txt` and the v2 prompt file `prompts/nemo_logic-artifacts-02.txt` SHALL remain on disk untouched. Deletion is deferred.

### FR-3 Few-shot examples
**FR-3.1** `extract_artifacts.py::SPAN_LEVEL_EXAMPLES` SHALL be the v2 `PAVEMENT_EXAMPLES` content unchanged: 3 `lx.data.ExampleData` entries totalling 25 extractions, all 21 span-level classes covered.

**FR-3.2** `extract_artifacts.py::CHUNK_LEVEL_EXAMPLES` SHALL contain at least 2 `lx.data.ExampleData` entries. Each entry SHALL contain at minimum:
- Exactly 1 `chunk_summary` extraction whose `extraction_text` is the last complete sentence of the entry's `text` (verifiable as a suffix substring).
- At least 2 `chunk_topic` extractions, with at least one carrying `attributes.topic_role="primary"` and at least one with `topic_role="secondary"`.
- At least 2 `pavement_engineering_term` extractions, each with `attributes.term`, `attributes.normalized_term`, and `attributes.term_category` populated.

**FR-3.3** Across `CHUNK_LEVEL_EXAMPLES`, all 3 chunk-level classes SHALL be demonstrated.

**FR-3.4** Source spans (`extraction_text`) in both example sets SHALL be verbatim substrings of the parent example `text` (no paraphrase, no whitespace normalization).

### FR-4 `LXConfig` field changes
**FR-4.1** `LXConfig.prompt_name` SHALL be removed (not aliased).
**FR-4.2** `LXConfig` SHALL gain `prompt_name_span: str = "nemo_logic-artifacts-03-span"`.
**FR-4.3** `LXConfig` SHALL gain `prompt_name_chunk: str = "nemo_logic-artifacts-03-chunk"`.
**FR-4.4** `LXConfig` SHALL gain `chunk_extraction_passes: int = 1`.
**FR-4.5** `LXConfig.extraction_passes` SHALL retain its v2 default (`2`) and SHALL apply to the span-level call only.
**FR-4.6** Other `LXConfig` fields (`model`, `api_key`, `temperature`, `max_char_buffer`, `prompt_lib`) SHALL retain their v2 defaults. No additional fields are added.

### FR-5 `lx.extract` call structure
**FR-5.1** `PavementExtractor.extract` SHALL call `lx.extract` exactly twice per chunk: once with the span-level prompt and `extraction_passes`, once with the chunk-level prompt and `chunk_extraction_passes`. Both calls SHALL use the same `text_or_documents`, `model_id`, `api_key`, `temperature`, `max_char_buffer`, and `show_progress=False`.

**FR-5.2** The span-level call SHALL pass `examples=SPAN_LEVEL_EXAMPLES`. The chunk-level call SHALL pass `examples=CHUNK_LEVEL_EXAMPLES`.

**FR-5.3** The span-level call SHALL run before the chunk-level call. Both calls SHALL be sequential (no `asyncio` / threading).

**FR-5.4** A `langextract` exception in either call SHALL log at `"NLP"`, emit `"extractions": {}` + `"error": "<message>"` for that chunk, and continue with the next chunk. (Identical failure-isolation semantics to v2; one error covers both calls.)

### FR-6 Schema and per-class transformation
**FR-6.1** Doc-level output shape SHALL be `{"doc_id": str, "artifacts": list}` — unchanged from v2.

**FR-6.2** Per-chunk wrapper SHALL be `{chunk_id, tokens, extractions, [error]}` — unchanged from v2.

**FR-6.3** `artifact_id` format SHALL remain `f"{doc_id}_chunk_{chunk_id}_art_{ext_idx}"`. The `ext_idx` counter SHALL be **continuous across both calls**, starting at 0 with the first emitted span-level extraction and continuing into the chunk-level extractions. Uniqueness within a chunk SHALL hold.

**FR-6.4** Idempotency: if `{doc_id}-logic-artifacts.json` exists and `--overwrite` is not set, the doc SHALL be skipped with a `"CHUNK"`-level cache-hit log line. Identical to v2.

**FR-6.5** Mode-3 guard at `_nemo.py:439` and inside `main()` / the `__main__` block SHALL remain unchanged.

**FR-6.6** Provider: `model="gpt-4o-mini"`, `api_key=os.getenv("OPENAI_API_KEY")` resolution. Unchanged.

**FR-6.7** **Span-level per-extraction shape**: each entry inside `extractions[<span-class>]` SHALL have keys `{artifact_id, text, description, significance, char_interval, attributes}`. `description` SHALL be a string (default `""` if the model omits). `significance` SHALL be `null` or a non-empty string (default `null` if the model omits). `attributes` SHALL contain only type-specific keys and the remaining common attributes; SHALL NOT contain `description` or `significance`. Identical to v2 FR-6.7.

**FR-6.8** **Chunk-level per-extraction shape**: each entry inside `extractions[<chunk-class>]` SHALL have keys `{artifact_id, text, char_interval, attributes}`. The keys `description` and `significance` SHALL NOT appear at top level. The `attributes` dict SHALL be passed through verbatim from `langextract` (no popping); type-specific keys (`summary`, `document_function`, `scope`, `topic`, `topic_role`, `term`, `normalized_term`, `term_category`) SHALL appear there as the model emits them.

**FR-6.9** **Class-group transformation contract**:
- Span-level transformation: copy `ext.attributes` to a mutable dict; pop `description` (default `""`) and `significance` (default `None`) from that dict; emit entry per FR-6.7.
- Chunk-level transformation: copy `ext.attributes` to a mutable dict (no popping); emit entry per FR-6.8.
- These transformations SHALL be the only places where the per-class shape is set; no other code path SHALL re-add or remove `description` / `significance` to / from the attributes dict.

### FR-7 `extract_artifacts.toml`
**FR-7.1** The `[langextract]` block SHALL set:
- `model = "gpt-4o-mini"` (unchanged)
- `temperature = 0.0` (unchanged)
- `extraction_passes = 2` (span-level; unchanged from v2)
- `chunk_extraction_passes = 1` (new)
- `max_char_buffer = 10000` (unchanged)
- `prompt_name_span = "nemo_logic-artifacts-03-span"` (new)
- `prompt_name_chunk = "nemo_logic-artifacts-03-chunk"` (new)
- `prompt_lib = "./prompts"` (unchanged)
**FR-7.2** The TOML key `prompt_name` SHALL NOT appear (removed; not aliased).
**FR-7.3** The `[paths]` section (`input_dir`) SHALL remain unchanged.
**FR-7.4** Comments in the file SHALL be updated where they reference the v2 defaults; the file SHALL still be loadable as TOML.

### FR-8 Documentation
**FR-8.1** `docs/qa-generation.md`'s Step 2 section SHALL keep the 21 span-level class rows and add a separate *Chunk-level vocabulary (3 classes)* subsection with rows for `chunk_summary`, `chunk_topic`, `pavement_engineering_term` and one-line definitions.
**FR-8.2** The schema JSON example in that section SHALL show **mixed output**: at least one span-level entry (e.g. `requirement`) demonstrating the v2 6-key shape and at least one chunk-level entry (`chunk_summary`) demonstrating the v3 4-key shape. The shape difference SHALL be explicitly noted.
**FR-8.3** The configuration code block SHALL reflect the FR-7.1 defaults.
**FR-8.4** A note SHALL state the quantity rules (exactly 1 `chunk_summary`; 1-5 `chunk_topic`; 0+ `pavement_engineering_term`) and the `chunk_summary` last-sentence span convention.
**FR-8.5** The cost note SHALL reflect ~3 model calls per chunk (span passes=2 + chunk passes=1) — roughly 1.5× the v2 per-chunk cost — at gpt-4o-mini input prices.

### FR-9 Non-interference and file preservation
**FR-9.1** The v1 prompt file `prompts/nemo_logic-artifacts.txt` and the v2 prompt file `prompts/nemo_logic-artifacts-02.txt` SHALL remain unmodified on disk.
**FR-9.2** No module or script outside the four files enumerated above (`extract_artifacts.py`, `extract_artifacts.toml`, the two new prompt files, `docs/qa-generation.md`) SHALL be edited by this feature.
**FR-9.3** Existing v2 `-logic-artifacts.json` files SHALL continue to be valid JSON; the script SHALL not corrupt them. Operators force-regenerate via `--overwrite`.

### FR-10 `chunk_summary` span convention
**FR-10.1** The chunk-level prompt SHALL instruct the model to set `extraction_text` for `chunk_summary` to the last complete sentence of the chunk; if the chunk ends mid-fragment (a list item, equation, or partial line), to the closest preceding complete sentence.
**FR-10.2** Each `chunk_summary` extraction in `CHUNK_LEVEL_EXAMPLES` SHALL demonstrate this convention: the `extraction_text` SHALL be the last complete sentence of the example's `text`.
**FR-10.3** The script SHALL NOT validate or enforce the convention at runtime. Compliance is observable via post-hoc inspection.

### FR-11 Quantity rules
**FR-11.1** The chunk-level prompt SHALL state: exactly 1 `chunk_summary` per chunk, 1-5 `chunk_topic` per chunk, 0+ `pavement_engineering_term` per chunk (only important domain terms).
**FR-11.2** The script SHALL NOT cap, drop, or de-duplicate chunk-level extractions based on these rules. All emitted extractions that pass the class gate (FR-1.5) SHALL be emitted to the output.
**FR-11.3** Drift from these rules SHALL be observable in post-hoc aggregate stats but SHALL NOT cause the script to fail.

---

## 4. Non-Functional Requirements

### NFR-1 Backward compatibility (file-level)
v2 `-logic-artifacts.json` files SHALL remain valid JSON; the script's idempotency gate (FR-6.4) skip-writes them. They are NOT structurally compatible with v3 outputs:
- The `extractions` keys differ: v2 has at most 21 span-class buckets; v3 has up to 24 buckets including chunk-level.
- The per-extraction shape inside chunk-level buckets is the 4-key shape (FR-6.8) which v2 lacks entirely.
- The span-level per-extraction shape inside span-class buckets is identical between v2 and v3.

Consumers reading both versions MUST dispatch on schema (presence of chunk-level bucket keys is sufficient; per-extraction-shape disambiguation is also possible). Operators regenerate v2 caches via `--overwrite`.

### NFR-2 Determinism
With `temperature=0.0`, repeated runs over identical input SHOULD produce identical output, modulo `langextract`'s internal retry / sampling behavior across `extraction_passes`. The script's bookkeeping (file walk order, per-doc loop, two-call sequence per chunk, `ext_idx` indexing) SHALL be deterministic.

### NFR-3 Observability
Every per-doc run SHALL emit a `"CHUNK"`-level summary or cache-hit line (unchanged from v2). Class-vocab drops at either gate (FR-1.4, FR-1.5) SHALL emit one `"NLP"` line per drop, identifying the call ("span call" or "chunk call"). Failed chunks SHALL emit one `"NLP"` line per failure (a single failure covers both calls per chunk). Cost telemetry remains deferred.

### NFR-4 Performance envelope
- Model calls per chunk: `extraction_passes + chunk_extraction_passes` = `2 + 1 = 3` (vs `2` in v2). +50% calls per chunk.
- `max_char_buffer` stays at 10000 for both calls; logical chunks (max ~8000 chars) are not sub-chunked. One LLM call per pass per logical chunk.
- Net cost per chunk vs v2: ~1.5× under default config. At gpt-4o-mini input prices, ~$0.0024/chunk.
- Latency per chunk: roughly doubles under sequential calls. Parallelization deferred.

### NFR-5 Schema invariant
The doc-level shape (`{doc_id, artifacts: [...]}`) and per-chunk wrapper (`{chunk_id, tokens, extractions, [error]}`) SHALL be strict invariants v2 → v3. The per-extraction shape varies by class group per FR-6.7 / FR-6.8. Schema-aware diffs between a v2 and v3 file for the same source chunk SHALL differ in: (a) presence of new chunk-level bucket keys (`chunk_summary`, `chunk_topic`, `pavement_engineering_term`) — additive in v3; (b) per-extraction key set inside chunk-level buckets — 4 keys in v3 vs not present in v2. Span-level buckets and their per-extraction shapes SHALL be byte-stable v2 → v3 modulo model nondeterminism.

### NFR-6 Dependency isolation
`extract_artifacts.py` SHALL remain free of `aisa.*` imports. (Carries forward unchanged from v1 / v2 NFRs.)

---

## 5. Interfaces

### 5.1 CLI interface
```text
python extract_artifacts.py [--cfg PATH] [--input_dir DIR] [--overwrite]
```
**Unchanged from v2.**

### 5.2 Python interface
```python
SPAN_LEVEL_CLASSES: list[str]                       # 21 elements (FR-1.1)
CHUNK_LEVEL_CLASSES: list[str]                      # 3 elements (FR-1.2)
EXTRACTION_CLASSES: list[str]                       # union, 24 (FR-1.3)

SPAN_LEVEL_EXAMPLES: list[lx.data.ExampleData]      # v2 PAVEMENT_EXAMPLES, renamed (FR-3.1)
CHUNK_LEVEL_EXAMPLES: list[lx.data.ExampleData]     # ≥2 entries, all 3 chunk classes (FR-3.2/3.3)

@dataclass
class LXConfig:
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    temperature: float = 0.0
    extraction_passes: int = 2
    chunk_extraction_passes: int = 1                # new
    max_char_buffer: int = 10000
    prompt_name_span: str = "nemo_logic-artifacts-03-span"   # new
    prompt_name_chunk: str = "nemo_logic-artifacts-03-chunk" # new
    prompt_lib: str = "./prompts"
    # prompt_name field removed (FR-4.1)

class PavementExtractor:
    def __init__(self, cfg: LXConfig) -> None: ...           # surface unchanged
    def extract(self, text: str, doc_id: str, chunk_id: int) -> dict[str, list[dict]]: ...

def main(cfg: dict, overwrite: bool = False) -> None: ...    # unchanged
```

### 5.3 File interface
Input: `{input_dir}/{doc_id}-logic-chunks.json` (unchanged from v2). Output: `{input_dir}/{doc_id}-logic-artifacts.json`. Schema:

```
{
  doc_id,
  artifacts: [
    {
      chunk_id, tokens,
      extractions: {
        # Span-level buckets — v2 6-key shape
        <span-class>: [
          {artifact_id, text, description, significance, char_interval, attributes}
        ],
        # Chunk-level buckets — v3 4-key shape
        <chunk-class>: [
          {artifact_id, text, char_interval, attributes}
        ]
      },
      [error]
    }
  ]
}
```

Per-extraction notes:
- Span-level: `description` is always present (string, possibly `""`); `significance` is `null` when not stated by the source; `attributes` contains type-specific + remaining common attrs (`subject`, `scope`, `context_reference`, `source_cue`).
- Chunk-level: no top-level `description` / `significance`; `attributes` contains type-specific keys (`summary`, `document_function`, `scope` for `chunk_summary`; `topic`, `topic_role` for `chunk_topic`; `term`, `normalized_term`, `term_category` for `pavement_engineering_term`).

### 5.4 Configuration interface
The `[langextract]` block in `extract_artifacts.toml` and (when applicable) `cfg/nemo.toml` SHALL match FR-7.1. The `[paths]` block (or `[general]` fallback) is unchanged.

### 5.5 Environment interface
- `OPENAI_API_KEY` — required. Resolution unchanged from v2.

### 5.6 Prompt interface
- File: `prompts/nemo_logic-artifacts-03-span.txt`. Plain text. Byte-equal to v2 prompt (FR-2.1). No `{placeholders}`.
- File: `prompts/nemo_logic-artifacts-03-chunk.txt`. Plain text. Chunk-level body per FR-2.2. No `{placeholders}`.

---

## 6. Acceptance Criteria

- **AC-1** `prompts/nemo_logic-artifacts-03-span.txt` exists and is byte-equal to `prompts/nemo_logic-artifacts-02.txt`. (Verifiable by `cmp` or hash compare.)
- **AC-2** `prompts/nemo_logic-artifacts-03-chunk.txt` exists and contains: the 3 chunk-level class names, the quantity rules (`exactly one`, `one to five`, `zero or more`), the last-sentence span convention text, the topic vocabulary, the term-category vocabulary, the strict source-grounding rule for terms.
- **AC-3** `python -c "from extract_artifacts import SPAN_LEVEL_CLASSES, CHUNK_LEVEL_CLASSES, EXTRACTION_CLASSES; ..."` confirms `len(SPAN_LEVEL_CLASSES)==21`, `len(CHUNK_LEVEL_CLASSES)==3`, `EXTRACTION_CLASSES == SPAN_LEVEL_CLASSES + CHUNK_LEVEL_CLASSES`, and the two subsets are disjoint.
- **AC-4** `SPAN_LEVEL_EXAMPLES` is byte-equal to v2 `PAVEMENT_EXAMPLES` (3 entries, 25 extractions, 21 distinct span classes covered).
- **AC-5** `CHUNK_LEVEL_EXAMPLES` has at least 2 entries. Each entry has exactly 1 `chunk_summary` whose `extraction_text` is the last complete sentence of the entry's `text` (verified as a suffix substring). At least one `chunk_topic` per entry has `topic_role="primary"` and at least one has `topic_role="secondary"`. At least one `pavement_engineering_term` per entry has all of `term`, `normalized_term`, `term_category` populated. All 3 chunk-level classes are demonstrated across the example set.
- **AC-6** `LXConfig()` has `prompt_name_span=="nemo_logic-artifacts-03-span"`, `prompt_name_chunk=="nemo_logic-artifacts-03-chunk"`, `extraction_passes==2`, `chunk_extraction_passes==1`. The instance has no `prompt_name` attribute.
- **AC-7** Cold run on the existing TBF000027 / TBF000011 / TBF000131 fixtures writes `-logic-artifacts.json` with the v3 schema: top-level `{doc_id, artifacts}`; per-chunk `{chunk_id, tokens, extractions, [error]}`. Span-level entries have keys `{artifact_id, text, description, significance, char_interval, attributes}`; chunk-level entries have keys `{artifact_id, text, char_interval, attributes}` exactly.
- **AC-8** Class isolation across the cold-run output: every emitted span-level call class ∈ `SPAN_LEVEL_CLASSES`; every emitted chunk-level call class ∈ `CHUNK_LEVEL_CLASSES`. No span class appears in a chunk-call output and vice versa. Verified by aggregator regex.
- **AC-9** Each chunk has `count(chunk_summary) ∈ {0, 1}` (1 expected) and `count(chunk_topic) ≤ 5` (information-only; failures logged but not blocking).
- **AC-10** All `artifact_id`s match `^.+_chunk_\d+_art_\d+$`; uniqueness within the output dir holds; the counter is continuous across the span+chunk calls per chunk (span-level extractions use indices `0..N-1`, chunk-level use `N..M`).
- **AC-11** Idempotency: second invocation with no `--overwrite` writes nothing (mtime unchanged) and makes zero `langextract` calls. Identical to v2.
- **AC-12** Mode-3 guard (`extract_artifacts.py` and `_nemo.py:439`) still rejects `recursive` and `logical` with the redirect error.
- **AC-13** `grep -E '^\s*(extraction_passes|chunk_extraction_passes|prompt_name_span|prompt_name_chunk)\s*=' extract_artifacts.toml` outputs the four FR-7.1 lines with the v3 values. `grep -E '^\s*prompt_name\s*=' extract_artifacts.toml` returns no match.
- **AC-14** `docs/qa-generation.md` Step 2 section: the class table contains 21 span-level rows + 3 chunk-level rows in a clearly-labelled subsection; the schema JSON example shows both a span-level entry (with `description` / `significance` at top level) and a chunk-level entry (without those keys); the configuration block matches FR-7.1; the `chunk_summary` last-sentence note and quantity-rule note are present.
- **AC-15** Per-extraction shape verification across the cold-run output: every span-level entry has `description` as a string (possibly `""`), `significance` is `null` or a non-empty string, and `attributes` contains no `description` / `significance` keys. Every chunk-level entry has no `description` / `significance` top-level keys and `attributes` is a dict.
- **AC-16** Spot-checks: at least one `chunk_summary` has non-empty `attributes.summary`; at least one `pavement_engineering_term` has populated `attributes.normalized_term` and `attributes.term_category`; at least one `chunk_topic` has `topic_role="primary"`.

---

## 7. Risks and Open Questions

### 7.1 Risks

- **R-1 Cross-mode class pollution.** The model in either call may emit a class belonging to the other group (e.g. the span-level call returning `chunk_summary`, or the chunk-level call returning `requirement`). Mitigation: per-call class gates (FR-1.4, FR-1.5) drop the offender with an `"NLP"` log line that names the call. Aggregate counts surface the rate.
- **R-2 Quantity-rule drift.** The model may emit 0 or 2+ `chunk_summary`, or >5 `chunk_topic`, despite the prompt rule. Mitigation: prompt-only enforcement; aggregate counts (AC-9) make drift visible. If drift is empirically painful, a soft validator (keep first `chunk_summary`, cap `chunk_topic` at 5, log NLP) is a clean follow-up (see OQ-2).
- **R-3 Term flooding.** The model may extract every technical-looking phrase as `pavement_engineering_term`. Mitigation: the prompt's "do not extract generic words" rule + few-shot density (~3 terms per chunk-level example, not 20). Per-chunk cap is a follow-up if real-corpus runs flood.
- **R-4 v2/v3 cache mixing.** A directory holding v2 `-logic-artifacts.json` files will skip-write under v3 (FR-6.4 idempotency). The two file shapes are not interchangeable for downstream consumers (NFR-1). Mitigation: documented in `docs/qa-generation.md`; visible from the cache-hit log line; force regeneration via `--overwrite`.
- **R-5 chunk_summary span overlap with span-level extractions.** Because `chunk_summary.extraction_text` is the last sentence of the chunk and span-level types like `best_practice` / `recommendation` often cluster at chunk endings, langextract's `WordAligner` will hand out `MATCH_FUZZY` warnings on whichever extraction the global block-matcher consumes second. Cosmetic — alignment offsets remain correct. The two calls are independent so the warnings come from intra-call alignment within each call (chunk-level call: `chunk_summary` span vs `chunk_topic` / `pavement_engineering_term` spans pulled from the same vicinity). Acceptable.
- **R-6 Term-multi-occurrence MATCH_FUZZY.** Chunk-level term extractions often have spans that appear many times (e.g. "RCA" in the corpus). The aligner picks one via fuzzy fallback. Tag is harmless — canonical term lives in `attributes.term` / `attributes.normalized_term`. Don't try to silence.
- **R-7 Performance.** Per-chunk wall time roughly doubles under sequential calls. Mitigation: parallel calls via threading / asyncio is a follow-up (OQ-1). For the current fixture-driven workflow, doubled latency is acceptable.

### 7.2 Open questions (non-blocking)

- **OQ-1 Parallel calls.** Can the span and chunk `lx.extract` calls run concurrently via `concurrent.futures.ThreadPoolExecutor`? `langextract==1.2.1` is sync; threads are fine because the underlying HTTP call releases the GIL. Worth a short benchmark once v3 is stable.
- **OQ-2 Code-side quantity validator.** If R-2 manifests, a soft validator that drops trailing `chunk_summary` entries beyond the first and caps `chunk_topic` at 5 (with `"NLP"` logs) is a clean follow-up. Not v3 scope.
- **OQ-3 Topic / term-category code canonicalizer.** If real-corpus runs show topic-vocabulary or term-category drift (model picking near-misses like "structural design" instead of "rigid pavement design"), a post-hoc canonicalizer can map drift to canonical labels. Defer until measured.
- **OQ-4 Schema versioning marker.** v2 SRS OQ-5 carries forward. v3 outputs are disambiguable from v2 by per-extraction shape and bucket keys; an explicit `"schema_version": 3` marker at the top level is cleaner for mixed-cache directories. Defer until a downstream consumer requests it.
- **OQ-5 Delete v1 / v2 prompt files.** Once v3 is validated on a real corpus, `prompts/nemo_logic-artifacts.txt` and `prompts/nemo_logic-artifacts-02.txt` become dead code. Cleanup PR.
- **OQ-6 Chunk-level few-shot expansion.** The minimum 2-entry `CHUNK_LEVEL_EXAMPLES` covers all 3 chunk classes but with thin demonstration breadth (~6-9 extractions per class across the set). Adding a 3rd entry covering a different document type (research-style report) is a likely follow-up if real-corpus chunk-level outputs lack diversity.
