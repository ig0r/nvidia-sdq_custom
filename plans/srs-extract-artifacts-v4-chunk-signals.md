# Software Requirements Specification: Extract-Artifacts v4 — Direct OpenAI Structured Outputs for Chunk Signals

**Feature:** Replace the v3 langextract-based chunk-level extraction (3 langextract classes via a second `lx.extract` call) with a direct OpenAI Structured Outputs call validated by Pydantic models. Span-level extraction (21 classes via `langextract`) is preserved verbatim from v3. The per-chunk wrapper in `-logic-artifacts.json` gains a new `chunk_signals` field carrying a single structured object (summary + 1-5 topics + 0+ terms); chunk-level entries no longer appear inside the `extractions` class-bucketed dict.
**Component:** `nvidia-sdq_custom`
**Version:** 0.4 (draft)
**Status:** Proposed
**Companion plan:** `plans/plan-extract-artifacts-v4-chunk-signals.md`

---

## 1. Introduction

### 1.1 Purpose
This SRS defines requirements for moving chunk-level extraction off `langextract` and onto a direct OpenAI Structured Outputs call validated by Pydantic. The goal is to eliminate v3's structural friction on chunk-level (artificial `extraction_text` spans, MATCH_FUZZY noise, prompt-only quantity rules, unnatural class-bucketed shape for what is conceptually a single structured object per chunk) while preserving span-level extraction unchanged. Span-level continues to use `langextract` because verbatim spans + alignment offsets are core to that pipeline's downstream value; chunk-level signals don't have those needs.

### 1.2 Scope

In scope:
- Two new prompt files: `prompts/nemo_logic-artifacts-04-span.txt` (byte-equal copy of v3 span prompt) and `prompts/nemo_logic-artifacts-04-chunk.txt` (new chunk-level system prompt for the OpenAI call, no langextract idioms).
- Removal of `CHUNK_LEVEL_CLASSES` and `CHUNK_LEVEL_EXAMPLES` from `extract_artifacts.py`. `EXTRACTION_CLASSES` reverts to `SPAN_LEVEL_CLASSES` (21 elements).
- Addition of Pydantic models in `extract_artifacts.py`: `ChunkSummary`, `ChunkTopic`, `PavementTerm`, `ChunkSignals`. Three `Literal` enum aliases: `DOCUMENT_FUNCTION`, `TOPIC_ROLE`, `TERM_CATEGORY`.
- Addition of `ChunkLevelExtractor` class wrapping `openai.OpenAI().beta.chat.completions.parse(response_format=ChunkSignals, …)`.
- `LXConfig` field changes: drop `prompt_name_span` (renamed to `prompt_name`), `prompt_name_chunk` (renamed to `chunk_prompt_name`), `chunk_extraction_passes` (removed; chunk call is single-pass deterministic).
- Refactor `PavementExtractor.extract` to orchestrate both calls with per-call failure isolation; return a unified per-chunk record `{extractions, chunk_signals, error}`.
- New `chunk_signals` top-level field on the per-chunk wrapper in `-logic-artifacts.json`.
- Updates to `extract_artifacts.toml` for the new field names.
- Updates to `docs/qa-generation.md` Step 2 section.

Out of scope:
- Provider swap. v3's OpenAI `gpt-4o-mini` and `OPENAI_API_KEY` resolution stay (user-confirmed across v1/v2/v3/v4).
- Doc-level wrapper (`{doc_id, artifacts: [...]}`) — unchanged.
- Span-level `artifact_id` format — unchanged. Chunk-level entries (topics, terms) carry no `artifact_id` in v4.
- Mode-3 guard, idempotency semantics — unchanged.
- `lx.extract` call signature for the span-level call — unchanged from v3.
- Span-level prompt body — `prompts/nemo_logic-artifacts-04-span.txt` is byte-equal to `prompts/nemo_logic-artifacts-03-span.txt`.
- `SPAN_LEVEL_EXAMPLES` content — byte-equal to v3's content (3 entries, 25 extractions, all 21 span-level classes).
- Code-side hard enforcement of `topics` array length (1-5). OpenAI Structured Outputs does not honor `minItems`/`maxItems`; the script soft-validates post-receipt with logging + cap-at-5.
- Code-side validation of `topic.topic` label drift against the canonical topic vocabulary (the vocabulary is in the prompt; Pydantic does not enforce it because `topic: str`).
- Code-side validation of `term.term` strictness (verbatim from source). Source-grounding is prompt-only.
- Few-shot example inside the chunk prompt. Schema + clear field semantics are the only steering mechanism in v4.
- Parallel span/chunk calls (asyncio / threading).
- Migration tooling for v3 outputs.
- Deletion of v1 / v2 / v3 prompt files.
- Schema-version marker at the doc level.
- Adding `char_interval` / `text` offsets for `pavement_engineering_term` entries (deferred — can be added with a `text.find()` post-pass if downstream needs).

### 1.3 Definitions
- **v3** — prior state, 24 langextract classes (21 span + 3 chunk), two `lx.extract` calls per chunk, chunk-level entries inside `extractions` class buckets.
- **v4** — state introduced by this SRS, 21 langextract classes (span only) + a structured `chunk_signals` object per chunk produced by a direct OpenAI Structured Outputs call.
- **Span-level extraction** — `langextract` call against the 21-class taxonomy; output bucketed in `extractions` per class.
- **Chunk-level signals** — single structured object per chunk produced by the OpenAI Structured Outputs call; output in the new `chunk_signals` field on the per-chunk wrapper.
- **Pydantic-enforced** — validated at parse time by Pydantic via OpenAI's `response_format=<Model>`. Includes: presence of required fields, scalar types (str / list), enum values for `Literal[...]`-typed fields. Excludes: array length constraints (OpenAI Structured Outputs ignores `minItems`/`maxItems`).
- **Soft-validated** — post-receipt validation in Python with logging and capping (not rejection of the whole call).

### 1.4 References
- `plans/plan-extract-artifacts-v4-chunk-signals.md` — companion implementation plan.
- `plans/plan-extract-artifacts-v3-chunk-level.md`, `plans/srs-extract-artifacts-v3-chunk-level.md` — v3 plan and SRS.
- OpenAI Structured Outputs documentation (`response_format={"type":"json_schema","strict":true,…}`); Python SDK `openai.OpenAI().beta.chat.completions.parse()` Pydantic-aware variant.
- Pydantic v2 `BaseModel` + `Literal[...]` enum support.

---

## 2. Overall Description

### 2.1 Product Perspective
v3 extracts chunk-level signals via a second `langextract` call against a 3-class taxonomy. The mismatch between langextract's verbatim-span-and-alignment design and the chunk-level use case (where the actual signal lives in `attributes.summary` / `attributes.topic` / `attributes.term` rather than in the span itself) produces friction: artificial `extraction_text` selection, MATCH_FUZZY noise, prompt-only quantity rules, and an unnatural class-bucketed output shape.

v4 keeps span-level extraction unchanged (langextract earns its keep on verbatim spans + alignment) and routes chunk-level through a direct OpenAI call with `response_format=ChunkSignals`. The chunk-level output becomes a single Pydantic-validated structured object surfaced in a new `chunk_signals` field on the per-chunk wrapper. Length constraints on the `topics` list are soft-validated; enum constraints on `document_functions`, `topic_role`, and `term.category` are Pydantic-enforced.

The doc-level wrapper (`{doc_id, artifacts: [...]}`) is byte-compatible v3 → v4. The per-chunk wrapper gains a `chunk_signals` field; the existing `extractions` field shape is unchanged structurally but no longer carries chunk-level class buckets.

### 2.2 User Classes
- **Pipeline operator** — runs `python extract_artifacts.py` after Step 1 to produce artifact files.
- **Pipeline developer** — extends or tunes the taxonomy, prompts, Pydantic models.
- **Downstream consumer** — reads `chunk_signals.summary.summary`, `chunk_signals.summary.document_functions`, `chunk_signals.topics`, `chunk_signals.terms` for retrieval, filtering, QA generation, entity linking.

### 2.3 Operating Environment
Identical to v3: Python 3.11+ (venv at 3.12), `langextract==1.2.1`, `openai==1.91.0`, `pydantic==2.11.7`, `loguru`, `python-dotenv`, `OPENAI_API_KEY` populated in `.env` or environment.

### 2.4 Constraints
- The 21-class span-level vocabulary SHALL be defined as `SPAN_LEVEL_CLASSES`. `EXTRACTION_CLASSES` SHALL equal `SPAN_LEVEL_CLASSES`. No chunk-level langextract classes.
- The chunk-level output SHALL conform to the `ChunkSignals` Pydantic model. The model SHALL be passed as `response_format` to OpenAI's Structured Outputs API.
- The doc-level shape (`{doc_id, artifacts: [...]}`) SHALL NOT change.
- The per-chunk wrapper SHALL be `{chunk_id, tokens, extractions, chunk_signals, [error]}`. `extractions` SHALL contain only span-level class buckets (subset of `SPAN_LEVEL_CLASSES`). `chunk_signals` SHALL be either a `ChunkSignals`-conformant dict or `null`.
- Span-level per-extraction shape SHALL be unchanged from v3: `{artifact_id, text, description, significance, char_interval, attributes}`. Chunk-level entries (topics, terms) SHALL NOT carry `artifact_id`, `text`, `char_interval`, `description`, or `significance` keys.
- The provider SHALL remain OpenAI `gpt-4o-mini` for both calls; `OPENAI_API_KEY` resolution unchanged.
- `lx.extract` call signature for the span-level call SHALL be the v3 shape: `text_or_documents`, `prompt_description`, `examples`, `model_id`, `api_key`, `temperature`, `extraction_passes`, `max_char_buffer=10000`, `show_progress=False`.
- The OpenAI chunk-level call SHALL use `client.beta.chat.completions.parse(model=..., temperature=..., messages=..., response_format=ChunkSignals)` exactly. No alternate API surfaces.
- Both prompt files SHALL be loaded the same way: via `Path(cfg.prompt_lib) / f"{cfg.<field>}.txt"`.
- Span call and chunk call SHALL be sequential (span first, then chunk).
- Chunk-call failures SHALL NOT abort span-call results, and vice versa. Each call is independently failure-isolated.

### 2.5 Assumptions
- `prompts/nemo_logic-artifacts-04-span.txt` will be created as a byte-equal copy of `prompts/nemo_logic-artifacts-03-span.txt`.
- `prompts/nemo_logic-artifacts-04-chunk.txt` will be created with the v4 chunk-level system prompt (no `extraction_text` rules; field-by-field semantics for `summary`, `document_functions`, `scope`, `topics`, `terms`; the v3 meta-phrase ban for `summary.summary`).
- `openai>=1.40.0` and `pydantic>=2.0.0` are available in the Python environment (`openai==1.91.0` and `pydantic==2.11.7` per current `reqs.txt`).
- The operator regenerates stale v3 outputs via `--overwrite` if v4 outputs are needed in a directory that already has v3 files.

---

## 3. Functional Requirements

### FR-1 Extraction-class vocabulary (v4)
**FR-1.1** `extract_artifacts.py::SPAN_LEVEL_CLASSES` SHALL be exactly the v3 21-class list (unchanged):
```python
[
    "requirement", "condition", "exception", "constraint",
    "procedure", "method", "formula", "parameter",
    "threshold", "definition", "actor_role", "deliverable",
    "assumption", "finding", "recommendation", "best_practice",
    "decision", "rationale", "issue", "risk", "evidence",
]
```
**FR-1.2** `EXTRACTION_CLASSES` SHALL equal `SPAN_LEVEL_CLASSES` (21 elements).
**FR-1.3** `CHUNK_LEVEL_CLASSES` SHALL NOT be exported by the module.
**FR-1.4** `CHUNK_LEVEL_EXAMPLES` SHALL NOT be exported by the module.
**FR-1.5** Any extraction returned by the span-level `lx.extract` call whose class is not in `SPAN_LEVEL_CLASSES` SHALL be dropped, and the script SHALL emit one `"NLP"`-level log line naming the doc_id, chunk_id, and offending class. Identical to v3 FR-1.4.

### FR-2 Prompt files
**FR-2.1** A new prompt SHALL exist at `prompts/nemo_logic-artifacts-04-span.txt`. Its contents SHALL be byte-equal to `prompts/nemo_logic-artifacts-03-span.txt`.
**FR-2.2** A new prompt SHALL exist at `prompts/nemo_logic-artifacts-04-chunk.txt`. It SHALL contain:
- A task description: produce a structured chunk-signals object describing the whole chunk.
- Field-by-field semantics for the `ChunkSignals` schema:
  - `summary.summary`: 1-2 sentence content statement, source-grounded; self-contained; meta-phrase ban (`the chunk`, `this passage`, etc.).
  - `summary.document_functions`: list of one or more roles the source plays. Canonical labels enumerated (matching `DOCUMENT_FUNCTION` enum).
  - `summary.scope`: where/when/to-what the source applies; may be `null`.
  - `topics[].topic`: concise normalized label. Canonical vocabulary listed (matching the recommended topic suggestions).
  - `topics[].role`: `primary` or `secondary` (one primary per chunk).
  - `terms[].term`: verbatim domain term from source.
  - `terms[].normalized_term`: canonical version when useful; may be `null`.
  - `terms[].category`: enum value from the canonical list (matching `TERM_CATEGORY`).
- Quantity guidance: 1-5 topics (1 primary, the rest secondary); 0+ terms; do not extract generic words unless part of a meaningful technical term.
- Source-grounding: extract only stated information; do not infer related terms; do not add unstated significance.
**FR-2.3** The chunk-level prompt SHALL contain no `{placeholders}`, no langextract idioms (no `extraction_text`, `extraction_class`, `lx.data.Extraction`), and no `chunk_summary` / `chunk_topic` / `pavement_engineering_term` references **as class names**. (References to `topics` / `terms` as schema fields are required.)
**FR-2.4** The v1, v2, and v3 prompt files SHALL remain on disk untouched.

### FR-3 Few-shot examples
**FR-3.1** `extract_artifacts.py::SPAN_LEVEL_EXAMPLES` SHALL contain the v3 content unchanged: 3 `lx.data.ExampleData` entries totalling 25 extractions, all 21 span-level classes covered.
**FR-3.2** No `CHUNK_LEVEL_EXAMPLES` SHALL be defined or exported. The chunk-level call relies solely on prompt + Pydantic schema for output steering.

### FR-4 `LXConfig` field changes
**FR-4.1** `LXConfig.prompt_name_span` SHALL be removed; replaced by `LXConfig.prompt_name: str = "nemo_logic-artifacts-04-span"`.
**FR-4.2** `LXConfig.prompt_name_chunk` SHALL be removed; replaced by `LXConfig.chunk_prompt_name: str = "nemo_logic-artifacts-04-chunk"`.
**FR-4.3** `LXConfig.chunk_extraction_passes` SHALL be removed.
**FR-4.4** `LXConfig.extraction_passes` SHALL retain its v3 default (`2`), scoped to the span-level langextract call.
**FR-4.5** Other `LXConfig` fields (`model`, `api_key`, `temperature`, `max_char_buffer`, `prompt_lib`) SHALL retain their v3 defaults.

### FR-5 Two-call structure
**FR-5.1** `PavementExtractor.extract(text, doc_id, chunk_id)` SHALL call:
1. The span-level langextract pipeline (one `lx.extract` call configured with `extraction_passes` recall passes), yielding a class-bucketed dict gated against `SPAN_LEVEL_CLASSES`.
2. `ChunkLevelExtractor.extract(text, doc_id, chunk_id)`, which calls `openai.OpenAI().beta.chat.completions.parse(model=cfg.model, temperature=cfg.temperature, messages=[{system: chunk_prompt}, {user: text}], response_format=ChunkSignals)`.
**FR-5.2** The two calls SHALL be sequential (span first, then chunk).
**FR-5.3** Each call SHALL be independently failure-isolated:
- A span-call exception SHALL log at `"NLP"` ("span: …"), set `extractions = {}` for the chunk, and continue to the chunk call.
- A chunk-call exception SHALL log at `"NLP"` ("chunk: …"), set `chunk_signals = null` for the chunk, and continue.
- Both can fail independently; the per-chunk `error` field SHALL accumulate `"span: <msg>"` and/or `"chunk: <msg>"` entries.

### FR-6 Schema and per-chunk wrapper
**FR-6.1** Doc-level output shape SHALL be `{"doc_id": str, "artifacts": list}` — unchanged.
**FR-6.2** Per-chunk wrapper SHALL be `{chunk_id, tokens, extractions, chunk_signals, [error]}`. The `chunk_signals` field SHALL always be present; it is either a dict conforming to `ChunkSignals.model_dump()` or `null`.
**FR-6.3** `extractions` SHALL contain **only** span-level class buckets (subset of `SPAN_LEVEL_CLASSES`). It SHALL NOT contain `chunk_summary`, `chunk_topic`, or `pavement_engineering_term` as keys.
**FR-6.4** **Span-level per-extraction shape** SHALL be unchanged from v3: `{artifact_id, text, description, significance, char_interval, attributes}`. The class-vocabulary gate, the description/significance promotion, and the `artifact_id` format (`{doc_id}_chunk_{chunk_id}_art_{ext_idx}` with `ext_idx` starting at 0 per chunk) SHALL be preserved.
**FR-6.5** **Chunk-level signals shape** SHALL be:
```
chunk_signals: {
  summary: {
    summary: str,                                          # 1-2 sentences
    document_functions: list[DOCUMENT_FUNCTION],           # >=1
    scope: str | null
  },
  topics: [
    {topic: str, role: TOPIC_ROLE},                        # length 1-5 expected
    ...
  ],
  terms: [
    {term: str, normalized_term: str | null, category: TERM_CATEGORY},
    ...
  ]
}
```
Chunk-level list-items SHALL NOT carry `artifact_id`, `text`, `char_interval`, `description`, or `significance` keys.
**FR-6.6** Idempotency: if `{doc_id}-logic-artifacts.json` exists and `--overwrite` is not set, the doc SHALL be skipped with a `"CHUNK"`-level cache-hit log line. Identical to v3.
**FR-6.7** Mode-3 guard at `_nemo.py:439` and inside `main()` / the `__main__` block SHALL remain unchanged.
**FR-6.8** Provider: `model="gpt-4o-mini"` for both calls; `api_key=os.getenv("OPENAI_API_KEY")` resolution. Unchanged from v3.

### FR-7 Pydantic models and enums
**FR-7.1** `extract_artifacts.py` SHALL define module-level `Literal[...]` aliases:
- `DOCUMENT_FUNCTION` — exactly the 17 values: `"requirement"`, `"procedure"`, `"design guidance"`, `"calculation guidance"`, `"definition"`, `"approval workflow"`, `"material guidance"`, `"construction guidance"`, `"testing guidance"`, `"maintenance guidance"`, `"finding"`, `"recommendation"`, `"rationale"`, `"issue"`, `"risk"`, `"evidence"`, `"example"`.
- `TOPIC_ROLE` — exactly: `"primary"`, `"secondary"`.
- `TERM_CATEGORY` — exactly the 14 values: `"traffic"`, `"pavement_type"`, `"design_parameter"`, `"material"`, `"layer"`, `"method"`, `"test_method"`, `"distress"`, `"construction"`, `"maintenance"`, `"organization"`, `"form"`, `"software"`, `"other"`.
**FR-7.2** `ChunkSummary`, `ChunkTopic`, `PavementTerm`, `ChunkSignals` SHALL be `pydantic.BaseModel` subclasses with the field types specified in FR-6.5.
**FR-7.3** `ChunkSignals` SHALL be passed as `response_format` to `client.beta.chat.completions.parse(...)`. The OpenAI Structured Outputs strict-mode contract SHALL hold (required fields populated; enum values validated; type checking).
**FR-7.4** Length constraints on `topics` and `terms` SHALL NOT be encoded in the Pydantic models (OpenAI Structured Outputs does not honor `minItems`/`maxItems`); they are soft-validated per FR-8.

### FR-8 Soft validation
**FR-8.1** `ChunkLevelExtractor.extract` SHALL validate `len(signals.topics)` after Pydantic parsing:
- `0` → log one `"NLP"` line ("0 chunk topics emitted; expected 1-5"); pass through unchanged.
- `1..5` → no log.
- `>5` → log one `"NLP"` line ("N chunk topics emitted; capping to first 5"); truncate `signals.topics` to `signals.topics[:5]`.
**FR-8.2** No soft validation SHALL be applied to `terms` count (rule is "0+", no upper cap).
**FR-8.3** No soft validation SHALL be applied to `document_functions` count beyond the Pydantic-enforced `>=1` (Pydantic raises if empty list is emitted with required field; in practice the model is unlikely to omit).

### FR-9 `extract_artifacts.toml`
**FR-9.1** The `[langextract]` block SHALL set:
- `model = "gpt-4o-mini"` (unchanged)
- `temperature = 0.0` (unchanged)
- `extraction_passes = 2` (span-level; unchanged from v3)
- `max_char_buffer = 10000` (unchanged)
- `prompt_name = "nemo_logic-artifacts-04-span"` (renamed from `prompt_name_span`)
- `chunk_prompt_name = "nemo_logic-artifacts-04-chunk"` (renamed from `prompt_name_chunk`)
- `prompt_lib = "./prompts"` (unchanged)
**FR-9.2** The TOML keys `prompt_name_span`, `prompt_name_chunk`, `chunk_extraction_passes` SHALL NOT appear (removed; not aliased).
**FR-9.3** The `[paths]` section (`input_dir`) SHALL remain unchanged.
**FR-9.4** Comments in the file SHALL be updated where they reference v3 defaults; the file SHALL still be loadable as TOML.

### FR-10 Documentation
**FR-10.1** `docs/qa-generation.md`'s Step 2 section SHALL keep the 21 span-level class rows. The chunk-level subsection table from v3 SHALL be replaced by a description of the `chunk_signals` field shape (summary, document_functions, scope, topics, terms) and per-field semantics.
**FR-10.2** The schema JSON example SHALL show **both** the `extractions` field (with at least one span-level entry) and the `chunk_signals` field (with summary, topics, and terms populated).
**FR-10.3** The configuration code block SHALL match FR-9.1.
**FR-10.4** A note SHALL state that the chunk-level call is direct OpenAI Structured Outputs (not langextract), that summary singularity and enum constraints are Pydantic-enforced, and that `topics` count is soft-validated (1-5 expected, capped at 5).
**FR-10.5** The cost note SHALL reflect v4: ~2 model calls per chunk under default config (`extraction_passes=2` for span ⇒ 2 langextract calls + 1 OpenAI call ≈ 2-3 model calls per chunk depending on internal langextract pass behavior). Chunk call is cheaper than v3 because no few-shot examples and shorter prompt.

### FR-11 Non-interference and file preservation
**FR-11.1** The v1, v2, and v3 prompt files SHALL remain unmodified on disk.
**FR-11.2** No module or script outside the four files enumerated above (`extract_artifacts.py`, `extract_artifacts.toml`, the two new prompt files, `docs/qa-generation.md`) SHALL be edited by this feature.
**FR-11.3** Existing v3 `-logic-artifacts.json` files SHALL continue to be valid JSON; the script SHALL not corrupt them. Operators force-regenerate via `--overwrite`.

---

## 4. Non-Functional Requirements

### NFR-1 Backward compatibility (file-level)
v3 `-logic-artifacts.json` files SHALL remain valid JSON; the script's idempotency gate (FR-6.6) skip-writes them. They are NOT structurally compatible with v4 outputs:
- v3 had chunk-level entries inside `extractions` under keys `chunk_summary` / `chunk_topic` / `pavement_engineering_term`. v4 has none of those keys in `extractions`; instead the per-chunk wrapper has a new top-level `chunk_signals` field.
- v3 chunk-level entries had a 4-key shape (`{artifact_id, text, char_interval, attributes}`). v4 chunk-level list-items have no `artifact_id` / `text` / `char_interval` keys; they have `topic`/`role` (topics) or `term`/`normalized_term`/`category` (terms) directly.
- Span-level entry shape SHALL be byte-stable v3 → v4 modulo model nondeterminism.

Consumers reading both versions MUST dispatch on schema (presence of the `chunk_signals` field is sufficient). Operators regenerate v3 caches via `--overwrite`.

### NFR-2 Determinism
With `temperature=0.0` on both calls, repeated runs over identical input SHOULD produce identical output, modulo OpenAI's internal nondeterminism (small variations in chat-completion sampling persist even at temperature 0). The script's bookkeeping (file walk order, per-doc loop, span-then-chunk call sequence, span `ext_idx` indexing) SHALL be deterministic.

### NFR-3 Observability
Every per-doc run SHALL emit a `"CHUNK"`-level summary or cache-hit line (unchanged from v3). Span-class drops SHALL emit one `"NLP"` line per drop (unchanged from v3). Soft-validation events for the chunk call (`topics` count == 0 or > 5) SHALL emit one `"NLP"` line per event (FR-8.1). Failed calls (span or chunk) SHALL emit one `"NLP"` line per failure with the `"span:"` or `"chunk:"` prefix (FR-5.3).

### NFR-4 Performance envelope
- Span call: same as v3 (2 langextract passes; ~3960 input tokens per call including 25 few-shot extractions).
- Chunk call: ~800-1000 input tokens per call (no few-shot examples; shorter system prompt). One call per chunk.
- Total per-chunk cost: ~$0.002 at gpt-4o-mini input prices, comparable to v3 or slightly lower.
- Wall-time per chunk: comparable to v3 (sequential span+chunk calls).

### NFR-5 Schema invariant
The doc-level shape (`{doc_id, artifacts: [...]}`) SHALL be a strict invariant v3 → v4. The per-chunk wrapper SHALL gain a `chunk_signals` field (additive); the existing `extractions` field SHALL be byte-stable for span-level content. The per-extraction shape inside span-level buckets SHALL be unchanged.

### NFR-6 Dependency isolation
`extract_artifacts.py` SHALL remain free of `aisa.*` imports (carries forward from prior versions). It MAY import `openai`, `pydantic`, `langextract`, `loguru`, `dotenv` directly.

---

## 5. Interfaces

### 5.1 CLI interface
```text
python extract_artifacts.py [--cfg PATH] [--input_dir DIR] [--overwrite]
```
**Unchanged from v3.**

### 5.2 Python interface
```python
SPAN_LEVEL_CLASSES: list[str]                       # 21 elements (FR-1.1)
EXTRACTION_CLASSES: list[str]                       # alias of SPAN_LEVEL_CLASSES (FR-1.2)

SPAN_LEVEL_EXAMPLES: list[lx.data.ExampleData]      # v3 content unchanged (FR-3.1)

DOCUMENT_FUNCTION = Literal[...]                    # 17 values (FR-7.1)
TOPIC_ROLE        = Literal["primary", "secondary"]
TERM_CATEGORY     = Literal[...]                    # 14 values (FR-7.1)

class ChunkSummary(BaseModel):
    summary: str
    document_functions: list[DOCUMENT_FUNCTION]
    scope: str | None = None

class ChunkTopic(BaseModel):
    topic: str
    role: TOPIC_ROLE

class PavementTerm(BaseModel):
    term: str
    normalized_term: str | None = None
    category: TERM_CATEGORY

class ChunkSignals(BaseModel):
    summary: ChunkSummary
    topics: list[ChunkTopic]
    terms: list[PavementTerm]

@dataclass
class LXConfig:
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    temperature: float = 0.0
    extraction_passes: int = 2                              # span-level recall passes
    max_char_buffer: int = 10000
    prompt_name: str = "nemo_logic-artifacts-04-span"
    chunk_prompt_name: str = "nemo_logic-artifacts-04-chunk"
    prompt_lib: str = "./prompts"

class ChunkLevelExtractor:
    def __init__(self, cfg: LXConfig) -> None: ...
    def extract(self, text: str, doc_id: str, chunk_id: int) -> ChunkSignals: ...

class PavementExtractor:
    def __init__(self, cfg: LXConfig) -> None: ...           # surface unchanged
    def extract(self, text: str, doc_id: str, chunk_id: int) -> dict: ...
        # returns {"extractions": dict[str, list[dict]], "chunk_signals": dict | None, "error": str | None}

def main(cfg: dict, overwrite: bool = False) -> None: ...    # unchanged surface
```

### 5.3 File interface
Input: `{input_dir}/{doc_id}-logic-chunks.json` (unchanged from v3). Output: `{input_dir}/{doc_id}-logic-artifacts.json`. Schema:

```
{
  doc_id,
  artifacts: [
    {
      chunk_id, tokens,
      extractions: {
        # Span-level buckets only — v3 6-key shape unchanged
        <span-class>: [
          {artifact_id, text, description, significance, char_interval, attributes}
        ]
      },
      chunk_signals: {
        summary: {
          summary: str,
          document_functions: list[DOCUMENT_FUNCTION],
          scope: str | null
        },
        topics: [{topic: str, role: TOPIC_ROLE}, ...],
        terms: [{term: str, normalized_term: str | null, category: TERM_CATEGORY}, ...]
      } | null,
      [error]
    }
  ]
}
```

Notes:
- `chunk_signals` is `null` when the chunk-level call failed.
- `extractions` is `{}` when the span-level call failed.
- `error` accumulates `"span: <msg>"` and/or `"chunk: <msg>"` separated by `"; "`.

### 5.4 Configuration interface
The `[langextract]` block in `extract_artifacts.toml` SHALL match FR-9.1. The `[paths]` block (or `[general]` fallback) is unchanged.

### 5.5 Environment interface
- `OPENAI_API_KEY` — required. Resolution unchanged from v3.

### 5.6 Prompt interface
- File: `prompts/nemo_logic-artifacts-04-span.txt`. Plain text. Byte-equal to `prompts/nemo_logic-artifacts-03-span.txt` (FR-2.1). No `{placeholders}`.
- File: `prompts/nemo_logic-artifacts-04-chunk.txt`. Plain text. v4 chunk-level body per FR-2.2. No `{placeholders}`. No langextract idioms (FR-2.3).

---

## 6. Acceptance Criteria

- **AC-1** `prompts/nemo_logic-artifacts-04-span.txt` exists and is byte-equal to `prompts/nemo_logic-artifacts-03-span.txt`. (Verifiable by `cmp` or hash compare.)
- **AC-2** `prompts/nemo_logic-artifacts-04-chunk.txt` exists and contains: task description, field-by-field semantics for `summary`, `document_functions`, `scope`, `topics`, `terms`, quantity guidance (1-5 topics, 0+ terms), source-grounding rule, the meta-phrase ban for `summary.summary`. The prompt does **not** contain the words `extraction_text`, `extraction_class`, or the v3 langextract class names `chunk_summary` / `chunk_topic` / `pavement_engineering_term` as class identifiers.
- **AC-3** `python -c "from extract_artifacts import SPAN_LEVEL_CLASSES, EXTRACTION_CLASSES; ..."` confirms `len(SPAN_LEVEL_CLASSES) == 21` and `EXTRACTION_CLASSES == SPAN_LEVEL_CLASSES`. `CHUNK_LEVEL_CLASSES` and `CHUNK_LEVEL_EXAMPLES` are not importable from the module.
- **AC-4** `SPAN_LEVEL_EXAMPLES` is byte-equal to v3's `SPAN_LEVEL_EXAMPLES` (3 entries, 25 extractions, 21 distinct span classes covered).
- **AC-5** Pydantic models `ChunkSummary`, `ChunkTopic`, `PavementTerm`, `ChunkSignals` import cleanly. Enum aliases `DOCUMENT_FUNCTION` (17 values), `TOPIC_ROLE` (2 values), `TERM_CATEGORY` (14 values) import. `ChunkSignals.model_json_schema()` includes the enum constraints under each field's `enum`.
- **AC-6** `LXConfig()` defaults: `prompt_name == "nemo_logic-artifacts-04-span"`, `chunk_prompt_name == "nemo_logic-artifacts-04-chunk"`, `extraction_passes == 2`, `max_char_buffer == 10000`. The instance has no `prompt_name_span`, `prompt_name_chunk`, or `chunk_extraction_passes` attribute.
- **AC-7** `ChunkLevelExtractor` instantiates with a valid `LXConfig`, loads the chunk prompt file, and exposes `extract(text, doc_id, chunk_id) -> ChunkSignals`.
- **AC-8** Cold run on the existing fixtures (`TBF000027`, `TBF000011`, `TBF000131`) writes `-logic-artifacts.json`. Per-chunk wrapper shape: `{chunk_id, tokens, extractions, chunk_signals, [error]}`. `extractions` keys are a subset of `SPAN_LEVEL_CLASSES`. `chunk_signals` is either a dict matching `ChunkSignals.model_dump()` or `null`.
- **AC-9** Span-level entries in `extractions` have keys `{artifact_id, text, description, significance, char_interval, attributes}` exactly. Chunk-level list-items in `chunk_signals.topics` have keys `{topic, role}` only. Chunk-level list-items in `chunk_signals.terms` have keys `{term, normalized_term, category}` only.
- **AC-10** Quantity sanity: every chunk has `chunk_signals.summary` as a single object (never list, never null when chunk call succeeded); `len(chunk_signals.topics) ∈ [1, 5]` (cap enforced by FR-8.1); `len(chunk_signals.terms) ≥ 0`.
- **AC-11** Class isolation: every key in `extractions` ∈ `SPAN_LEVEL_CLASSES`. No `chunk_summary` / `chunk_topic` / `pavement_engineering_term` keys appear anywhere.
- **AC-12** All span-level `artifact_id`s match `^.+_chunk_\d+_art_\d+$`; uniqueness within the output dir holds. Chunk-level list-items SHALL NOT have `artifact_id` keys.
- **AC-13** Idempotency: second invocation with no `--overwrite` writes nothing (mtime unchanged) and makes zero langextract / OpenAI calls.
- **AC-14** Mode-3 guard rejects `recursive` and `logical`.
- **AC-15** TOML grep: `prompt_name`, `chunk_prompt_name`, `extraction_passes` present with v4 values; `prompt_name_span`, `prompt_name_chunk`, `chunk_extraction_passes` not present (`grep -E '^\s*(prompt_name_span|prompt_name_chunk|chunk_extraction_passes)\s*=' extract_artifacts.toml` returns no match).
- **AC-16** `docs/qa-generation.md` Step 2: 21 span-level class rows kept; the chunk-level subsection describes `chunk_signals` as a structured field (not as classes); schema JSON example shows both `extractions` and `chunk_signals`; configuration block matches FR-9.1; cost note matches FR-10.5; OpenAI Structured Outputs is named.
- **AC-17** Pydantic enum coverage in cold-run output: every `summary.document_functions[i]` ∈ `DOCUMENT_FUNCTION` enum values; every `topics[i].role` ∈ `TOPIC_ROLE`; every `terms[i].category` ∈ `TERM_CATEGORY`. (Pydantic raises `ValidationError` if the model emits invalid values, which surfaces as a `"chunk: ValidationError(...)"` error in the per-chunk `error` field.)
- **AC-18** Meta-phrase ban: no `chunk_signals.summary.summary` value across the cold-run output begins with or contains the meta-phrases `the chunk`, `this chunk`, `the text`, `this passage`, `this section`, `the document`, `the passage describes`, or `this excerpt` (case-insensitive). Verified by aggregator regex.

Spot-checks:
- At least one chunk has `chunk_signals.summary.document_functions ⊇ {"requirement"}` or another canonical role.
- At least one `topics` list contains a `primary`-roled topic.
- At least one `terms` entry has `normalized_term` populated and `category` ∈ the canonical 14.

---

## 7. Risks and Open Questions

### 7.1 Risks

- **R-1 OpenAI Structured Outputs API drift.** OpenAI may evolve the `response_format` schema requirements or `client.beta.chat.completions.parse()` interface. Mitigation: pin `openai==1.91.0` (already pinned); SDK migrations are routine. The Pydantic model abstraction insulates us from low-level JSON schema changes.
- **R-2 Pydantic validation failures.** The model may emit a value outside the `DOCUMENT_FUNCTION` / `TOPIC_ROLE` / `TERM_CATEGORY` enums. OpenAI strict-mode is supposed to prevent this, but edge cases exist. Mitigation: failure is caught and surfaced as `"chunk: ValidationError(...)"` in the per-chunk error field; span call still produces output. Aggregate stats reveal failure rate.
- **R-3 Quantity drift on `topics`.** Length constraints aren't enforced at schema level. Model may emit 0 topics (unusable) or 6+ (capped). Mitigation: soft-validation in `ChunkLevelExtractor.extract` (FR-8.1) logs and caps. Empirical drift surfaces in `"NLP"` log aggregation.
- **R-4 Term flooding.** With no few-shot example, the model may extract every technical-looking phrase as a term. Mitigation: prompt's "do not extract generic words" rule + explicit category enum (forces categorization, which discourages generic terms). If empirical flooding occurs, an inline JSON example in the prompt is a clean follow-up.
- **R-5 v3/v4 cache mixing.** A directory holding v3 `-logic-artifacts.json` files will skip-write under v4 (FR-6.6 idempotency). The two file shapes are not interchangeable for downstream consumers (NFR-1). Mitigation: documented in `docs/qa-generation.md`; visible from the cache-hit log line; force regeneration via `--overwrite`.
- **R-6 Loss of `pavement_engineering_term` source offsets.** v3 had `char_interval` on each term; v4 doesn't. If downstream needs offsets (e.g. for highlighting in a UI), they're missing. Mitigation: a 5-line `text.find()` post-pass in `ChunkLevelExtractor.extract` can add `start_pos`/`end_pos` to each term. Not v4 scope; defer until a downstream consumer asks.
- **R-7 No `topic` vocabulary enforcement.** `topic.topic` is `str`, not enum-typed. Model may emit non-canonical labels (`"structural design"` instead of `"rigid pavement design"`). Mitigation: prompt enumerates canonical labels; aggregate analysis surfaces drift; a post-hoc canonicalizer is a clean follow-up if needed.
- **R-8 Two SDK paths to maintain.** v4 uses both `langextract` and the `openai` SDK directly. Mitigation: the surfaces are well-isolated (`PavementExtractor` for span, `ChunkLevelExtractor` for chunk); each can be tuned independently.

### 7.2 Open questions (non-blocking)

- **OQ-1 Inline few-shot example in chunk prompt.** Schema enforcement + clear field semantics may suffice. If real-corpus runs show level-of-detail or term-selection drift, add 1 inline JSON example at the end of the chunk prompt (token cost ~200, negligible). Defer until measured.
- **OQ-2 Schema-version marker at the doc level.** Per-chunk wrapper shape (`chunk_signals` field presence) disambiguates v3 vs v4. An explicit `"schema_version": 4` top-level marker is more robust for mixed-cache directories. Defer until a downstream consumer requests it.
- **OQ-3 Char offsets for terms.** Adding `text.find()` lookups on each term to populate `start_pos` / `end_pos`. Defer until a downstream consumer asks.
- **OQ-4 Parallel span/chunk calls.** Both calls are sync but I/O-bound. `concurrent.futures.ThreadPoolExecutor` could halve per-chunk wall time. Defer until throughput is a felt constraint.
- **OQ-5 Topic vocabulary canonicalizer.** Post-hoc map drifted topic labels to canonical ones. Defer until aggregate runs reveal drift.
- **OQ-6 Delete v1 / v2 / v3 prompt files.** Once v4 is validated on a real corpus, the older prompts become dead code. Cleanup PR.
- **OQ-7 Stable per-item ids in `chunk_signals`.** Currently no `id` on topics/terms; downstream identifies by chunk_id + array index. If QA generation references specific topics ("which term was used in the answer?"), stable ids would help. Ids could be derived (`f"{doc_id}_chunk_{cid}_topic_{i}"`) without storing. Defer until a consumer needs them.
- **OQ-8 Refactor span-level out of `langextract`.** Some operators may eventually want span-level on Structured Outputs too (lose `char_interval` but gain enum-typed attribute keys). Out of scope for v4; revisit if span-level shows similar friction.
