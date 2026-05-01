# Plan: Extract-Artifacts v4 — Direct OpenAI Structured Outputs for Chunk Signals

## Context

v3 introduced chunk-level extraction on top of v2's 21-class span-level taxonomy by adding three new langextract classes (`chunk_summary`, `chunk_topic`, `pavement_engineering_term`) emitted via a second `lx.extract` call per chunk. The implementation works, but the design has structural friction:

1. **Artificial spans.** `chunk_summary.extraction_text` is forced to be the chunk's last sentence by convention, because langextract requires verbatim source spans for every emitted extraction. The actual signal is `attributes.summary` (a generated 1-2-sentence overview); the span is filler.
2. **Wasted alignment.** langextract's `WordAligner` produces a `char_interval` for every chunk-level extraction, but these offsets are not load-bearing for the summary or for topic spans (only `pavement_engineering_term` benefits).
3. **MATCH_FUZZY noise.** 12 of 14 example chunk-level extractions align as `MATCH_FUZZY` because the long `chunk_summary` block consumes the matcher first; the noise is cosmetic but persistent.
4. **Quantity rules unenforceable.** "Exactly 1 `chunk_summary`, 1-5 `chunk_topic`" is prompt-only steering — the model can drift to 0 or 2+.
5. **Unnatural output shape.** Chunk-level signals are conceptually a *single structured object per chunk* (one summary + a list of topics + a list of terms), but langextract forces three flat class-bucketed lists.

For span-level extraction these tradeoffs are correct: every span-level artifact *is* a clause from the source, alignment matters, and recall via `extraction_passes` is genuine value. For chunk-level, langextract's distinctive features (verbatim text, alignment, multi-pass recall) are unused or counterproductive.

v4 keeps span-level on langextract unchanged and routes chunk-level through a direct OpenAI Structured Outputs call. The chunk-level output becomes a single Pydantic-validated structured object per chunk, surfaced as a new `chunk_signals` field on the per-chunk wrapper alongside the existing `extractions` dict.

## Scope

**In scope**
- New prompt file `prompts/nemo_logic-artifacts-04-span.txt` — byte-equal copy of `prompts/nemo_logic-artifacts-03-span.txt`. Naming keeps the v4 prompt set coherent.
- New prompt file `prompts/nemo_logic-artifacts-04-chunk.txt` — chunk-level system prompt for the OpenAI call. No `extraction_text` rules; no last-sentence span convention; describes only the structured-output schema and the per-field semantics (summary, document_functions, scope, topics, terms).
- `EXTRACTION_CLASSES` reverts to `SPAN_LEVEL_CLASSES` (21 elements). `CHUNK_LEVEL_CLASSES` is removed; chunk-level types are no longer langextract classes.
- `CHUNK_LEVEL_EXAMPLES` is removed from `extract_artifacts.py`. `SPAN_LEVEL_EXAMPLES` retains its v3 content (3 entries, 25 extractions, all 21 span classes).
- New Pydantic models in `extract_artifacts.py`: `ChunkSummary`, `ChunkTopic`, `PavementTerm`, `ChunkSignals`. Type-specific enums exposed as Python `Literal[...]` aliases: `DOCUMENT_FUNCTION`, `TOPIC_ROLE`, `TERM_CATEGORY`.
- New `ChunkLevelExtractor` class wrapping `openai.OpenAI().beta.chat.completions.parse(response_format=ChunkSignals, …)`. Returns a validated `ChunkSignals` object per chunk.
- `LXConfig` reshaped: drop `prompt_name_chunk` and `chunk_extraction_passes`; add `chunk_prompt_name` (default `"nemo_logic-artifacts-04-chunk"`). The chunk model + temperature reuse the existing fields by default; advanced operators can override the chunk prompt only.
- `PavementExtractor.extract` returns a unified per-chunk record `{"extractions": <span-bucketed-dict>, "chunk_signals": <dict|None>, "errors": {"span": <str|None>, "chunk": <str|None>}}` instead of just the bucketed dict. Per-call failure isolation populates the corresponding `errors.<call>` slot.
- Per-chunk wrapper in `-logic-artifacts.json` gains two top-level fields: `chunk_signals` and `errors` (always-present dict with `span` and `chunk` keys). Span-level `extractions` shape is unchanged from v3.
- `extract_artifacts.toml` `[artifact_extraction]` block (renamed from v3's `[langextract]`): drop `chunk_extraction_passes`, rename `prompt_name_span` → `prompt_name`, add `chunk_prompt_name`.
- `docs/qa-generation.md` Step 2 section updated.
- `reqs.txt` — no change needed (`openai==1.91.0` and `pydantic==2.11.7` already pinned via langextract's transitive deps; we use them directly now).

**Out of scope**
- Provider switch — keep OpenAI `gpt-4o-mini`, `OPENAI_API_KEY` resolution.
- Doc-level wrapper (`{doc_id, artifacts: [...]}`) — unchanged from v3.
- `artifact_id` format for span-level — unchanged. Chunk-level entries have **no** `artifact_id` in v4 (chunk_id + array index identifies them; downstream can derive stable ids if needed).
- Mode-3 guard, idempotency, `_resolve_input_dir` — unchanged.
- Span-level prompt body, span-level few-shot examples, span-level transformation — unchanged.
- Code-side hard enforcement of `topics` count (1-5). Pydantic enforces *type* (list of `ChunkTopic`); array length is post-hoc validated with a soft cap (truncate + `"NLP"` log if >5; warn but accept if 0).
- Parallel span/chunk calls (asyncio / threading). Sequential first; parallelization is a follow-up.
- Migration tooling for v3 outputs — operator regenerates with `--overwrite`.
- Deletion of v1 / v2 / v3 prompt files — kept on disk for reference.
- Schema-version marker at the doc level — still deferred (per-chunk shape disambiguates v3 vs v4: v4 has `chunk_signals` field, v3 does not).
- Inline few-shot example inside the v4 chunk prompt. Start with prompt+schema only; add an inline JSON example later if quality issues emerge.

## Per-chunk extraction flow (v4)

```mermaid
sequenceDiagram
    autonumber
    participant Caller as main()
    participant PE as PavementExtractor
    participant LX as langextract
    participant CE as ChunkLevelExtractor
    participant OAI as OpenAI SDK

    Caller->>PE: extract(text, doc_id, chunk_id)
    Note over PE,LX: Span-level call (extraction_passes=2)
    PE->>LX: lx.extract(span prompt, SPAN_LEVEL_EXAMPLES)
    alt span call succeeds
        LX-->>PE: extractions
        PE->>PE: gate vs SPAN_LEVEL_CLASSES; promote description/significance
    else span call fails
        LX-->>PE: exception
        PE->>PE: errors["span"] = str(exc)
    end
    Note over PE,OAI: Chunk-level call (single pass)
    PE->>CE: extract(text, doc_id, chunk_id)
    CE->>OAI: parse(response_format=ChunkSignals)
    alt chunk call succeeds
        OAI-->>CE: ChunkSignals (Pydantic-validated)
        CE->>CE: soft-cap topics at 5
        CE-->>PE: ChunkSignals
        PE->>PE: model_dump()
    else chunk call fails
        OAI-->>CE: exception
        CE-->>PE: re-raise
        PE->>PE: errors["chunk"] = str(exc)
    end
    PE-->>Caller: {extractions, chunk_signals, errors}
```

The two calls are independent — either can fail without aborting the other. `errors` is always present as a dict with `span` and `chunk` keys (each `null` on success).

## Concrete changes

### New: `prompts/nemo_logic-artifacts-04-span.txt`
Byte-equal copy of `prompts/nemo_logic-artifacts-03-span.txt`. v4 set is coherent (`-04-span` + `-04-chunk`).

### New: `prompts/nemo_logic-artifacts-04-chunk.txt`
Chunk-level system prompt for the OpenAI Structured Outputs call. Plain text, no langextract idioms, no `extraction_text`/`extraction_class` references. Sections:
- **Task**: produce a structured chunk-signals object describing the whole chunk.
- **Field semantics**:
  - `summary.summary`: 1-2 sentence content statement, source-grounded; *self-contained* (no "the chunk", "this passage", etc. — the v3 meta-phrase ban carries forward).
  - `summary.document_functions`: list of one or more roles the source plays (canonical labels listed; enum-typed downstream).
  - `summary.scope`: where/when/to-what the source applies, if stated.
  - `topics[].topic`: concise normalized label (canonical vocabulary listed).
  - `topics[].role`: `primary` (one) or `secondary` (rest).
  - `terms[].term`: verbatim domain term from the source.
  - `terms[].normalized_term`: canonical version when useful.
  - `terms[].category`: enum value from the canonical list.
- **Quantity guidance**: emit 1-5 topics (1 primary, the rest secondary); 0+ terms — only important domain terms; do not extract generic words (`pavement`, `design`, `project`, `section`, `material`) unless part of a meaningful technical term.
- **Source grounding**: extract only information stated in the source. Do not infer related terms. Do not add unstated significance.

### Modify: `extract_artifacts.py`

**Vocabulary** — chunk-level classes removed:
```python
SPAN_LEVEL_CLASSES: list[str] = [
    "requirement", "condition", "exception", "constraint",
    "procedure", "method", "formula", "parameter",
    "threshold", "definition", "actor_role", "deliverable",
    "assumption", "finding", "recommendation", "best_practice",
    "decision", "rationale", "issue", "risk", "evidence",
]
EXTRACTION_CLASSES: list[str] = SPAN_LEVEL_CLASSES   # 21
# CHUNK_LEVEL_CLASSES removed
```

**Examples** — `SPAN_LEVEL_EXAMPLES` unchanged from v3 (3 entries, 25 extractions). `CHUNK_LEVEL_EXAMPLES` removed.

**Pydantic models + enums** — new module-level definitions:
```python
from pydantic import BaseModel, Field
from typing import Literal

DOCUMENT_FUNCTION = Literal[
    "requirement", "procedure", "design guidance", "calculation guidance",
    "definition", "approval workflow", "material guidance", "construction guidance",
    "testing guidance", "maintenance guidance", "finding", "recommendation",
    "rationale", "issue", "risk", "evidence", "example",
]
TOPIC_ROLE = Literal["primary", "secondary"]
TERM_CATEGORY = Literal[
    "traffic", "pavement_type", "design_parameter", "material", "layer",
    "method", "test_method", "distress", "construction", "maintenance",
    "organization", "form", "software", "other",
]

class ChunkSummary(BaseModel):
    summary: str = Field(description="1-2 sentence content statement, source-grounded, self-contained")
    document_functions: list[DOCUMENT_FUNCTION] = Field(description="one or more roles the source plays")
    scope: str | None = Field(default=None, description="where/when/to what the source applies, if stated")

class ChunkTopic(BaseModel):
    topic: str = Field(description="concise normalized topic label")
    role: TOPIC_ROLE

class PavementTerm(BaseModel):
    term: str = Field(description="verbatim domain term from source")
    normalized_term: str | None = Field(default=None, description="canonical version, if useful")
    category: TERM_CATEGORY

class ChunkSignals(BaseModel):
    summary: ChunkSummary
    topics: list[ChunkTopic]   # quantity 1-5, soft-validated post-receipt
    terms: list[PavementTerm]  # 0+
```

**`LXConfig`** — simplified:
```python
@dataclass
class LXConfig:
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    temperature: float = 0.0
    extraction_passes: int = 2                            # span-level recall
    max_char_buffer: int = 10000
    prompt_name: str = "nemo_logic-artifacts-04-span"
    chunk_prompt_name: str = "nemo_logic-artifacts-04-chunk"
    prompt_lib: str = "./prompts"
```
Removed: `prompt_name_span` (renamed to `prompt_name`), `prompt_name_chunk` (renamed `chunk_prompt_name`), `chunk_extraction_passes` (no concept of "passes" for a single deterministic OpenAI call).

**`ChunkLevelExtractor`** — new class:
```python
class ChunkLevelExtractor:
    def __init__(self, cfg: LXConfig):
        self.cfg = cfg
        api_key = cfg.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api_key)
        prompt_path = Path(cfg.prompt_lib) / f"{cfg.chunk_prompt_name}.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Chunk prompt not found: {prompt_path}")
        self.system_prompt = prompt_path.read_text(encoding="utf-8")

    def extract(self, text: str, doc_id: str, chunk_id: int) -> ChunkSignals:
        completion = self.client.beta.chat.completions.parse(
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
            response_format=ChunkSignals,
        )
        signals = completion.choices[0].message.parsed
        # Soft validation of topics quantity rule
        n = len(signals.topics)
        if n == 0:
            logger.log("NLP", f"{doc_id} chunk {chunk_id}: 0 chunk topics emitted (expected 1-5)")
        elif n > 5:
            logger.log("NLP", f"{doc_id} chunk {chunk_id}: {n} chunk topics emitted; capping to first 5")
            signals.topics = signals.topics[:5]
        return signals
```

**`PavementExtractor.extract`** — orchestrates both calls and returns a unified per-chunk record:
```python
def extract(self, text: str, doc_id: str, chunk_id: int) -> dict:
    span_extractions: dict = {}
    chunk_signals: dict | None = None
    errors: dict[str, str | None] = {"span": None, "chunk": None}
    # Span-level call (langextract; unchanged from v3)
    try:
        span_extractions = self._extract_spans(text, doc_id, chunk_id)
    except Exception as exc:
        logger.log("NLP", f"{doc_id} chunk {chunk_id}: span extraction failed: {exc}")
        errors["span"] = str(exc)
    # Chunk-level call (OpenAI Structured Outputs)
    try:
        signals = self.chunk_extractor.extract(text, doc_id, chunk_id)
        chunk_signals = signals.model_dump(exclude_none=False)
    except Exception as exc:
        logger.log("NLP", f"{doc_id} chunk {chunk_id}: chunk-signals extraction failed: {exc}")
        errors["chunk"] = str(exc)
    return {
        "extractions": span_extractions,
        "chunk_signals": chunk_signals,
        "errors": errors,
    }
```

The span-extraction body (`_extract_spans`) is the v3 span-only path verbatim: one langextract call against `SPAN_LEVEL_EXAMPLES` and the v4 span prompt, gated against `SPAN_LEVEL_CLASSES`, with description/significance promotion. `ext_idx` resets to 0 per chunk (no continuous-counter-across-calls anymore — only span-level extractions get `artifact_id`).

**`main()`** — minor adjustment to write the new shape:
```python
artifacts.append({
    "chunk_id": chunk_id,
    "tokens": tokens,
    "extractions": result["extractions"],
    "chunk_signals": result["chunk_signals"],
    "errors": result["errors"],
})
```
`errors` is always present; both keys (`span`, `chunk`) are `null` on full success.

### Modify: `extract_artifacts.toml`
```toml
[artifact_extraction]
model = "gpt-4o-mini"
temperature = 0.0
extraction_passes = 2                       # span-level recall passes
max_char_buffer = 10000
prompt_name = "nemo_logic-artifacts-04-span"
chunk_prompt_name = "nemo_logic-artifacts-04-chunk"
prompt_lib = "./prompts"
# api_key resolved from .env::OPENAI_API_KEY at runtime if absent here
```
Removed: `chunk_extraction_passes`, `prompt_name_span`, `prompt_name_chunk`. **Section header renamed**: v3's `[langextract]` becomes `[artifact_extraction]` — v3's name was accurate when both calls went through langextract, but with the chunk call moved to direct OpenAI Structured Outputs, the old name was misleading. No alias kept; clean break. Operators upgrading from v3 must rename the section header in their config.

### Modify: `docs/qa-generation.md`
- Class table: keep the 21 span-level rows. Remove the chunk-level subsection table (3 rows). Replace with a description of the `chunk_signals` field shape.
- Schema example: shows `extractions` (span-level only, v3-compatible) **and** the new `chunk_signals` field with a populated example.
- Configuration block: matches the new TOML keys.
- Cost note: 2 model calls per chunk (1 langextract span × 2 passes ≈ 2 calls + 1 OpenAI Structured Outputs ≈ 1 call); chunk call is cheaper than v3 because no examples and shorter prompt; net per-chunk cost roughly equal to v3.
- One-line note that the chunk prompt is silent on `extraction_text` / spans / quantity rules at schema level (Pydantic enforces summary-singularity and field types; soft cap applies to topics list).

### Unchanged
- `extract_artifacts.py` plumbing: argparse, `_resolve_input_dir`, idempotency, `--overwrite`, output file path.
- `cfg/nemo.toml` — backward compat via the input-dir glob path still works.
- `_nemo.py:439` mode-3 guard.
- `reqs.txt` — `openai==1.91.0` and `pydantic==2.11.7` already pinned.
- v1/v2/v3 prompt files — kept on disk for reference.

## Behavior notes

- **Per-chunk wrapper grows two fields.** Top-level shape (`{doc_id, artifacts: [...]}`) unchanged. Per-chunk wrapper now `{chunk_id, tokens, extractions, chunk_signals, errors}`. All four new/changed fields are always present.
- **`chunk_signals` is a dict, not a list.** It contains a single summary, a list of 1-5 topics, and 0+ terms. On chunk-level failure, `chunk_signals` is `null`. On span failure, `extractions` is `{}`. Both can fail independently.
- **`errors` is always a dict** with keys `span` and `chunk`. Each value is `null` on success or an exception message on failure. Consumers dispatch on which key is non-null without parsing prefixes.
- **No `artifact_id` on chunk-level entries.** Topics and terms are list-items inside `chunk_signals` — identified by chunk_id + array index. If downstream wants stable ids, derive them: `f"{doc_id}_chunk_{cid}_topic_{i}"`. Span-level `artifact_id` format is unchanged.
- **No `extraction_text`, no spans, no `char_interval` for chunk-level.** Topics and terms have no source offsets; they're conceptual labels and verbatim strings respectively. If `pavement_engineering_term` offsets are needed downstream, a 5-line `text.find()` post-pass can add them as a follow-up.
- **Pydantic validation**: `summary` is exactly one object (schema-enforced); `document_functions` is `list[DOCUMENT_FUNCTION]` enum-typed; `topic_role` and `category` are enum-typed. **Length constraints on `topics` and `terms` are NOT enforced at schema level** (OpenAI Structured Outputs ignores `minItems`/`maxItems`); soft validation post-receipt logs and caps `topics` at 5.
- **Cost**: span call ~3960 tokens × 2 passes ≈ 7920 tokens (v3 unchanged). Chunk call drops from v3's ~3960 tokens × 1 pass to v4's ~800-1000 tokens (no few-shot examples; shorter prompt). Net per-chunk cost ≈ v3 or slightly lower. ~$0.002/chunk at gpt-4o-mini input prices.
- **Latency**: two sequential calls, similar to v3. Span call dominates.
- **MATCH_FUZZY noise drops**: 12 chunk-level fuzzy warnings per chunk-level call → 0 (chunk call no longer goes through langextract).
- **Cache invalidation**: existing v3 `-logic-artifacts.json` files lack the `chunk_signals` field and have chunk-level entries inside `extractions` instead. They remain valid JSON; v4 skip-writes them. Operators regenerate with `--overwrite`.

## Verification

Re-run on the existing fixtures with `--overwrite`:

```bash
.venv/bin/python extract_artifacts.py --cfg extract_artifacts.toml --overwrite
```

AC harness (extending v3):
- **AC v4-1** `prompts/nemo_logic-artifacts-04-span.txt` exists and is byte-equal to `prompts/nemo_logic-artifacts-03-span.txt`.
- **AC v4-2** `prompts/nemo_logic-artifacts-04-chunk.txt` exists and contains: task description, field-by-field semantics for `summary`/`document_functions`/`scope`/`topics`/`terms`, quantity guidance (1-5 topics, 0+ terms), source-grounding rule, the meta-phrase ban for `summary.summary`. **No** mention of `extraction_text`, `extraction_class`, `chunk_summary` (as a class), `chunk_topic` (as a class), or `pavement_engineering_term` (as a class).
- **AC v4-3** `EXTRACTION_CLASSES` is exactly the 21 span-level classes; equals `SPAN_LEVEL_CLASSES`. `CHUNK_LEVEL_CLASSES` is not exported. `CHUNK_LEVEL_EXAMPLES` is not exported.
- **AC v4-4** `SPAN_LEVEL_EXAMPLES` content is byte-equal to v3's `SPAN_LEVEL_EXAMPLES` (3 entries, 25 extractions).
- **AC v4-5** Pydantic models import cleanly: `ChunkSummary`, `ChunkTopic`, `PavementTerm`, `ChunkSignals`. Enum aliases `DOCUMENT_FUNCTION`, `TOPIC_ROLE`, `TERM_CATEGORY` import. `ChunkSignals.model_json_schema()` includes the enum constraints.
- **AC v4-6** `LXConfig()` defaults: `prompt_name="nemo_logic-artifacts-04-span"`, `chunk_prompt_name="nemo_logic-artifacts-04-chunk"`, `extraction_passes=2`, `max_char_buffer=10000`. No `prompt_name_span` / `prompt_name_chunk` / `chunk_extraction_passes` fields.
- **AC v4-7** Cold run on the fixtures writes `-logic-artifacts.json`. Per-chunk wrapper shape: `{chunk_id, tokens, extractions, chunk_signals, errors}` — all four mandatory. `extractions` is span-level only (no `chunk_summary` / `chunk_topic` / `pavement_engineering_term` buckets). `chunk_signals` is a dict matching the `ChunkSignals` Pydantic schema, or `null` on chunk-call failure. `errors` is always `{"span": <str|null>, "chunk": <str|null>}`.
- **AC v4-8** Span-level entries have keys `{artifact_id, text, description, significance, char_interval, attributes}` (unchanged from v3). Chunk-level entries inside `chunk_signals.topics[]` and `chunk_signals.terms[]` have **no** `artifact_id` / `text` / `char_interval` / `description` / `significance` keys.
- **AC v4-9** Quantity sanity: `chunk_signals.summary` is always a single object (Pydantic-enforced); `len(chunk_signals.topics) ∈ [1, 5]` (1 expected via prompt; 5 cap enforced by `ChunkLevelExtractor`); `len(chunk_signals.terms) ≥ 0`.
- **AC v4-10** Class isolation: every key in `extractions` ∈ `SPAN_LEVEL_CLASSES`. No span-level extraction has a chunk-level class name.
- **AC v4-11** All span-level `artifact_id`s match `^.+_chunk_\d+_art_\d+$`; uniqueness within the output dir holds.
- **AC v4-12** Idempotency: second invocation with no `--overwrite` writes nothing and makes zero langextract / OpenAI calls.
- **AC v4-13** Mode-3 guard rejects `recursive` and `logical`.
- **AC v4-14** TOML section: `[artifact_extraction]` header present; `[langextract]` header not present. Inside `[artifact_extraction]`: `prompt_name`, `chunk_prompt_name`, `extraction_passes` present with v4 values; `prompt_name_span`, `prompt_name_chunk`, `chunk_extraction_passes` not present.
- **AC v4-15** `docs/qa-generation.md` Step 2: 21 span-level class rows; `chunk_signals` field described (not as a class, as a structured field on the wrapper); schema example shows both `extractions` and `chunk_signals`; configuration block matches v4 TOML.
- **AC v4-16** Pydantic enum coverage: `summary.document_functions` values ∈ DOCUMENT_FUNCTION enum; `topics[].role` ∈ TOPIC_ROLE enum; `terms[].category` ∈ TERM_CATEGORY enum. (Pydantic raises `ValidationError` if model emits invalid values; failure surfaces in chunk-call error.)

Spot-checks:
- At least one chunk has non-empty `chunk_signals.summary.summary`.
- At least one chunk has `chunk_signals.summary.document_functions` ⊇ `{"requirement"}` or similar.
- At least one `topics` list contains a `primary` role.
- At least one `terms` entry has a populated `normalized_term`.
- No `chunk_signals.summary.summary` value contains the meta-phrases (`the chunk`, `this passage`, etc.).

## Decisions flagged

- **Chunk-level off langextract; span-level stays.** langextract earns its keep where verbatim spans + alignment matter (span-level). For chunk-level signals, schema enforcement via OpenAI Structured Outputs is a strictly better fit: enum-typed categorical fields, Pydantic-enforced summary singularity, no artificial spans, no MATCH_FUZZY noise.
- **Separate `chunk_signals` field, not flattened into `extractions`.** Chunk-level signals are conceptually one structured object per chunk, not a class-bucketed list. Forcing them into `extractions` would obscure the natural shape. Trade-off: per-chunk wrapper grows by one field; consumers reading both v3 and v4 dispatch on field presence.
- **No `artifact_id` for chunk-level entries.** Topics and terms are list-items inside `chunk_signals`; chunk_id + array index identifies them. Eliminates the v3 continuous-counter-across-calls complication and removes a non-load-bearing id namespace. Downstream can derive stable ids if needed.
- **`document_functions: list[DOCUMENT_FUNCTION]` not single Literal.** A chunk often plays multiple roles (e.g. requirement + approval workflow). List-of-enum lets the model express that without invented free-text combinations.
- **Soft cap on `topics` (5), not hard reject.** OpenAI Structured Outputs ignores `minItems`/`maxItems`. We cap and log at receipt time rather than reject the whole call. Quantity drift is observable, not fatal.
- **No few-shot example in the chunk prompt initially.** Schema enforcement + clear field semantics replace the example-driven format steering. If real-corpus runs show level-of-detail or term-selection drift, an inline JSON example is a clean follow-up.
- **`LXConfig` field renames clean up v3's awkward `_span` / `_chunk` suffixes.** v4 has `prompt_name` (the span prompt) and `chunk_prompt_name` (the chunk prompt). Asymmetric but aligns with span being the "default" extraction and chunk being the structured supplement.
- **`error` (string) → `errors` (dict) shape change.** v3 had a single optional `error` string covering whichever call failed. v4 has a required `errors` dict with two keys (`span`, `chunk`), each `null` on success or a string message on failure. The dict shape lets consumers dispatch on which call failed without parsing prefixes; both keys always present makes the schema regular.
- **TOML section renamed `[langextract]` → `[artifact_extraction]`.** v3's name was accurate when both calls went through langextract. With the chunk call moved to direct OpenAI Structured Outputs, the v3 name misled. v4's `[artifact_extraction]` describes what the section configures (the artifact-extraction step) without coupling to a specific library. Clean break, no alias.
- **Pydantic for chunk-level response shape, not raw JSON schema.** OpenAI's Structured Outputs API accepts both a Pydantic `BaseModel` (via `client.beta.chat.completions.parse()`) and a hand-written JSON schema dict (via `response_format={"type":"json_schema", ...}`). v4 uses Pydantic because: (a) `openai==1.91.0` already requires Pydantic (no new dep); (b) `.parse()` does schema generation, API call, parsing, and validation in one method; (c) single source of truth — the Pydantic class defines both the Python type and the JSON schema; (d) `Field(description=...)` flows into the schema as model-visible documentation; (e) `Literal[...]` enums are validated server-side via OpenAI strict mode and at parse time via Pydantic; (f) `pydantic.ValidationError` carries the exact field path on malformed output. The fallback to raw JSON schema is a ~10-line change isolated to `ChunkLevelExtractor`; reversal is cheap if `.parse()` is ever deprecated.
- **Old prompt files kept.** v1 / v2 / v3 prompts untouched on disk. Cleanup deferred until v4 validated on a real corpus.
- **No schema-version marker at the doc level.** Per-chunk wrapper shape (`chunk_signals` field presence) disambiguates v3 vs v4 in mixed-cache directories.
