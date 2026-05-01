# Plan: Extract-Artifacts v3 — Chunk-Level Artifacts

## Context

v2 produces normatively-typed span-level artifacts via a single langextract call per chunk against a 21-class taxonomy. Real-corpus runs (TBF000011, TBF000131) showed two structural gaps that span-level extraction can't close on its own:

1. **No chunk-wide signals.** Downstream retrieval / filtering / QA generation wants to know what a *whole chunk* says (a one-sentence summary), what it's about (1-5 normalized topics with primary/secondary roles), and which canonical pavement-engineering terms it anchors. A flat list of span-level artifacts under-serves these tasks; aggregating them at consumer time is duplicative work.
2. **Per-artifact attribute redundancy** if we tried to fix (1) at span level. An earlier proposal added `topic` / `pavement_engineering_terms` as common attributes on every span-level artifact. That repeats the same topic on 8-15 artifacts per chunk and gives no place to land a chunk summary or a typed term taxonomy. Discarded in favour of (1).

v3 closes the gap by adding three **chunk-level** classes (`chunk_summary`, `chunk_topic`, `pavement_engineering_term`) extracted via a *separate* langextract call per chunk against a dedicated chunk-level prompt and example set. Span-level extraction behaviour is unchanged from v2.

## Scope

**In scope**
- New prompt file `prompts/nemo_logic-artifacts-03-span.txt` — byte-equal copy of v2's `nemo_logic-artifacts-02.txt`, renamed to keep the v3 prompt set coherent.
- New prompt file `prompts/nemo_logic-artifacts-03-chunk.txt` — chunk-level instructions, quantity rules, span convention, topic vocabulary, term-category vocabulary.
- `EXTRACTION_CLASSES` extended from 21 → 24. Two named subsets exposed: `SPAN_LEVEL_CLASSES` and `CHUNK_LEVEL_CLASSES`.
- `PAVEMENT_EXAMPLES` renamed to `SPAN_LEVEL_EXAMPLES` (content unchanged from v2). New `CHUNK_LEVEL_EXAMPLES` few-shot set (2 entries, ~14 extractions, all 3 chunk-level classes covered, last-sentence span convention demonstrated).
- `LXConfig` gains `prompt_name_span`, `prompt_name_chunk`, `chunk_extraction_passes`. The `prompt_name` field is removed; `extraction_passes` now applies to span-level only.
- `PavementExtractor.extract` makes two `lx.extract` calls per chunk (span + chunk), gated against `SPAN_LEVEL_CLASSES` and `CHUNK_LEVEL_CLASSES` respectively. Results merged into a single class-bucketed dict with one continuous `artifact_id` counter.
- **Per-extraction shape varies by class group**: span-level entries keep v2 shape `{artifact_id, text, description, significance, char_interval, attributes}`; chunk-level entries use `{artifact_id, text, char_interval, attributes}` — `description` and `significance` are *not* promoted to top level for chunk-level types (they aren't part of the chunk-level schema).
- `extract_artifacts.toml` `[langextract]` updated to the new field names.
- `docs/qa-generation.md` Step 2 section updated.

**Out of scope**
- Provider switch — keep OpenAI `gpt-4o-mini`, `OPENAI_API_KEY` resolution.
- Doc-level wrapper, per-chunk wrapper, `artifact_id` format, mode-3 guard, idempotency, failure-isolation — unchanged.
- Code-side enforcement of quantity rules (exactly 1 `chunk_summary`, 1-5 `chunk_topic`) — prompt-only.
- Code-side validation of topic / term-category vocabularies — prompt-only.
- Parallel `lx.extract` calls (asyncio / threading) — sequential first; parallelization is a later optimization.
- Migration tooling for v2 outputs — operator regenerates with `--overwrite`.
- Deletion of v1 / v2 prompt files — kept on disk for reference.
- Schema-version marker at the doc level — v2 SRS OQ-5 still deferred; per-extraction shape disambiguates v2 vs v3.

## Concrete changes

### New: `prompts/nemo_logic-artifacts-03-span.txt`
Byte-equal copy of `prompts/nemo_logic-artifacts-02.txt`. Naming makes the v3 set coherent (`-03-span` + `-03-chunk`) so operators pin both at once.

### New: `prompts/nemo_logic-artifacts-03-chunk.txt`
Chunk-level prompt body covering:
- The 3 allowed classes with one-line definitions.
- Quantity rules (exactly 1 `chunk_summary`; 1-5 `chunk_topic`; 0+ important `pavement_engineering_term`).
- The `chunk_summary` span convention: *"Set extraction_text to the last complete sentence of the chunk. If the chunk ends mid-fragment, use the closest preceding complete sentence."*
- Per-class attribute schemas (`chunk_summary`: `summary`, `document_function`, `scope`; `chunk_topic`: `topic`, `topic_role`; `pavement_engineering_term`: `term`, `normalized_term`, `term_category`).
- Topic vocabulary suggestions (rigid/flexible pavement design, traffic loading, drainage, materials, construction, maintenance, rehabilitation, approval workflow, etc.).
- Term-category vocabulary (traffic, pavement_type, design_parameter, material, layer, method, test_method, distress, construction, maintenance, organization, form, software, other).
- Strict source-grounding rule for terms (no inferred or related terms; no generic words like "pavement", "design", "project" unless part of a meaningful technical term).
- No `description` / `significance` mentions — those keys are span-level only and the chunk-level prompt stays silent on them so the model doesn't emit them.

### Modify: `extract_artifacts.py`

**Vocabulary:**
```python
SPAN_LEVEL_CLASSES: list[str] = [
    "requirement", "condition", "exception", "constraint",
    "procedure", "method", "formula", "parameter",
    "threshold", "definition", "actor_role", "deliverable",
    "assumption", "finding", "recommendation", "best_practice",
    "decision", "rationale", "issue", "risk", "evidence",
]
CHUNK_LEVEL_CLASSES: list[str] = [
    "chunk_summary", "chunk_topic", "pavement_engineering_term",
]
EXTRACTION_CLASSES: list[str] = SPAN_LEVEL_CLASSES + CHUNK_LEVEL_CLASSES  # 24
```

**Examples:**
- `SPAN_LEVEL_EXAMPLES` — rename of v2 `PAVEMENT_EXAMPLES`. Content unchanged: 3 entries, 25 extractions, all 21 span classes covered.
- `CHUNK_LEVEL_EXAMPLES` — new. 2 entries:
  - **Example A** (manual/spec) — uses the RCA mix-design paragraph from `SPAN_LEVEL_EXAMPLES[1]` as its `text`. Demonstrates: 1 `chunk_summary` whose `extraction_text` is the last sentence of that paragraph; 3 `chunk_topic` entries (primary + 2 secondary); 3 `pavement_engineering_term` entries (`RCA`, `w/cm`, `28-day flexural strength`) with `term`, `normalized_term`, `term_category`.
  - **Example B** (report) — uses the RCA + brick-street findings paragraph from `SPAN_LEVEL_EXAMPLES[2]`. Demonstrates: 1 `chunk_summary` (last sentence span), 3 `chunk_topic`, 3 `pavement_engineering_term`.

**`LXConfig`:**
```python
@dataclass
class LXConfig:
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    temperature: float = 0.0
    extraction_passes: int = 2                            # span-level recall
    chunk_extraction_passes: int = 1                      # chunk-level (deterministic)
    max_char_buffer: int = 10000
    prompt_name_span: str = "nemo_logic-artifacts-03-span"
    prompt_name_chunk: str = "nemo_logic-artifacts-03-chunk"
    prompt_lib: str = "./prompts"
```
Removed: `prompt_name`. No alias kept — clean break.

**`PavementExtractor.__init__`** — load both prompt files; `FileNotFoundError` references both paths if either is missing.

**`PavementExtractor.extract`** — two-call structure:
```python
def extract(self, text, doc_id, chunk_id):
    # Span-level call
    span_result = lx.extract(
        text_or_documents=text,
        prompt_description=self.prompt_span,
        examples=SPAN_LEVEL_EXAMPLES,
        model_id=cfg.model, api_key=self.api_key,
        temperature=cfg.temperature,
        extraction_passes=cfg.extraction_passes,
        max_char_buffer=cfg.max_char_buffer,
        show_progress=False,
    )
    # Chunk-level call
    chunk_result = lx.extract(
        text_or_documents=text,
        prompt_description=self.prompt_chunk,
        examples=CHUNK_LEVEL_EXAMPLES,
        model_id=cfg.model, api_key=self.api_key,
        temperature=cfg.temperature,
        extraction_passes=cfg.chunk_extraction_passes,
        max_char_buffer=cfg.max_char_buffer,
        show_progress=False,
    )
    bucketed: dict[str, list[dict]] = {}
    ext_idx = 0
    # Span-level: gate against SPAN_LEVEL_CLASSES; promote description/significance.
    for ext in (span_result.extractions or []):
        cls = ext.extraction_class
        if cls not in SPAN_LEVEL_CLASSES:
            logger.log("NLP", f"{doc_id} chunk {chunk_id}: span call dropped class {cls!r}")
            continue
        attrs = dict(ext.attributes or {})
        description = attrs.pop("description", "") or ""
        significance = attrs.pop("significance", None) or None
        bucketed.setdefault(cls, []).append({
            "artifact_id": f"{doc_id}_chunk_{chunk_id}_art_{ext_idx}",
            "text": ext.extraction_text,
            "description": description,
            "significance": significance,
            "char_interval": _char_iv(ext),
            "attributes": attrs,
        })
        ext_idx += 1
    # Chunk-level: gate against CHUNK_LEVEL_CLASSES; bypass description/significance.
    for ext in (chunk_result.extractions or []):
        cls = ext.extraction_class
        if cls not in CHUNK_LEVEL_CLASSES:
            logger.log("NLP", f"{doc_id} chunk {chunk_id}: chunk call dropped class {cls!r}")
            continue
        bucketed.setdefault(cls, []).append({
            "artifact_id": f"{doc_id}_chunk_{chunk_id}_art_{ext_idx}",
            "text": ext.extraction_text,
            "char_interval": _char_iv(ext),
            "attributes": dict(ext.attributes or {}),
        })
        ext_idx += 1
    return bucketed
```

The `ext_idx` counter is **continuous across both calls** (span-level first, then chunk-level), so `artifact_id` uniqueness within a chunk holds and the v2 regex `^.+_chunk_\d+_art_\d+$` still passes.

### Modify: `extract_artifacts.toml`
```toml
[langextract]
model = "gpt-4o-mini"
temperature = 0.0
extraction_passes = 2                          # span-level recall passes
chunk_extraction_passes = 1                    # chunk-level passes (deterministic)
max_char_buffer = 10000
prompt_name_span = "nemo_logic-artifacts-03-span"
prompt_name_chunk = "nemo_logic-artifacts-03-chunk"
prompt_lib = "./prompts"
# api_key resolved from .env::OPENAI_API_KEY at runtime if absent here
```
Removed: `prompt_name`.

### Modify: `docs/qa-generation.md`
- Class table: keep the 21 span-level rows; add a clearly-labelled *Chunk-level vocabulary (3 classes)* subsection with rows for `chunk_summary`, `chunk_topic`, `pavement_engineering_term` and one-line definitions.
- Schema example: show **mixed output** — one `requirement` (span-level, with `description` / `significance` at top level) and one `chunk_summary` (chunk-level, without those keys, with `summary` / `document_function` / `scope` inside `attributes`). Highlight the per-class shape difference.
- Add a one-line note on quantity rules (exactly 1 `chunk_summary`; 1-5 `chunk_topic`; 0+ `pavement_engineering_term`) and the `chunk_summary` last-sentence span convention.
- Configuration block: update to v3 field names. Cost note: ~3 model calls per chunk (span passes=2 + chunk passes=1) vs 2 in v2; ~$0.0024/chunk at gpt-4o-mini input prices.

### Unchanged
- `extract_artifacts.py` plumbing: argparse, `_resolve_input_dir`, idempotency, failure isolation, output writing.
- `cfg/nemo.toml` — backward compat via the input-dir glob path still works.
- `_nemo.py:439` mode-3 guard.
- `reqs.txt` — no new dependencies.
- v1 prompt (`nemo_logic-artifacts.txt`) and v2 prompt (`nemo_logic-artifacts-02.txt`) stay on disk.

## Behavior notes

- **`chunk_summary` span = last sentence of chunk** (Option 2 with `[-1]`). Predictable, model-friendly, demonstrable in examples. Trade-off: span-level types like `best_practice` / `recommendation` often cluster at chunk endings, so MATCH_FUZZY warnings shift to that side. Cosmetic — alignment offsets remain correct.
- **Per-class shape difference is intentional.** Span-level entries have 6 top-level keys; chunk-level entries have 4. Consumers using `entry.get("description", "")` are unaffected. Schema-aware tools must dispatch on class group.
- **Continuous `artifact_id` counter** across both calls. Span-level extractions get the lower indices (0..N-1), chunk-level the higher (N..M). v2 id regex unchanged.
- **Cross-mode class drops**: if the span-level call somehow emits `chunk_summary`, or the chunk-level call emits `requirement`, the per-call class gate drops it with an `"NLP"` log. This catches model confusion across the two prompts.
- **No code-side quantity enforcement.** If the model emits 0 or 2+ `chunk_summary`, or >5 `chunk_topic`, all are kept. Aggregate counts in post-hoc analysis surface drift; tightening is a follow-up.
- **Term-multi-occurrence MATCH_FUZZY**: chunk-level `pavement_engineering_term` extractions often have spans appearing many times in the source (e.g. "RCA"). The aligner picks one via fuzzy fallback. Tag is harmless — the canonical term lives in `attributes.term` / `attributes.normalized_term`, not in the span. Don't try to silence.
- **Cost**: ~3 model calls per chunk (was 2 in v2). At gpt-4o-mini input prices: ~$0.0024/chunk. For a 100-chunk corpus, ~$0.24 vs ~$0.12 in v2. Negligible.
- **Latency**: sequential calls double per-chunk wall time. Throughput is corpus-dependent; not blocking for fixture-driven debugging.
- **Cache invalidation**: existing v2 `-logic-artifacts.json` files lack chunk-level buckets and use the span-only per-extraction shape. They remain valid JSON; v3 skip-writes them. Operators regenerate with `--overwrite`.

## Verification

Re-run on the existing TBF000011 / TBF000131 fixtures with `--overwrite`:

```bash
.venv/bin/python extract_artifacts.py --cfg extract_artifacts.toml --overwrite
```

AC harness (extending v2):
- **AC v3-1** `prompts/nemo_logic-artifacts-03-span.txt` exists and is byte-equal to `prompts/nemo_logic-artifacts-02.txt`.
- **AC v3-2** `prompts/nemo_logic-artifacts-03-chunk.txt` exists and contains: the 3 chunk-level class definitions, quantity rules, the last-sentence span convention, the topic vocabulary, the term-category vocabulary, the strict source-grounding rule for terms.
- **AC v3-3** `EXTRACTION_CLASSES` is exactly 24 elements: the 21-class v2 list followed by `chunk_summary`, `chunk_topic`, `pavement_engineering_term`. `SPAN_LEVEL_CLASSES` and `CHUNK_LEVEL_CLASSES` are exposed and disjoint.
- **AC v3-4** `SPAN_LEVEL_EXAMPLES` content is byte-equal to v2 `PAVEMENT_EXAMPLES`. `CHUNK_LEVEL_EXAMPLES` has 2 entries; each has exactly 1 `chunk_summary` whose `extraction_text` is the last sentence of its parent `text` (verifiable as a suffix substring).
- **AC v3-5** `LXConfig()` defaults: `prompt_name_span="nemo_logic-artifacts-03-span"`, `prompt_name_chunk="nemo_logic-artifacts-03-chunk"`, `extraction_passes=2`, `chunk_extraction_passes=1`. No `prompt_name` field.
- **AC v3-6** Cold run writes `-logic-artifacts.json`. Doc-level shape `{doc_id, artifacts}` unchanged. Per-chunk wrapper `{chunk_id, tokens, extractions, [error]}` unchanged.
- **AC v3-7** Per-extraction shape by class group:
  - span-level: keys `{artifact_id, text, description, significance, char_interval, attributes}`; `description` is a string; `significance` is `null` or non-empty string; `attributes` carries no `description` / `significance` keys.
  - chunk-level: keys `{artifact_id, text, char_interval, attributes}`; no `description` / `significance` at top level.
- **AC v3-8** Class isolation: every span-level call output class ∈ `SPAN_LEVEL_CLASSES`; every chunk-level call output class ∈ `CHUNK_LEVEL_CLASSES`. No `requirement` in chunk output, no `chunk_summary` in span output.
- **AC v3-9** Quantity sanity (informational; failures logged, not blocking): each chunk has `count(chunk_summary) ∈ {0, 1}` (1 expected) and `count(chunk_topic) ≤ 5`.
- **AC v3-10** `artifact_id` uniqueness within the output dir holds; counter is continuous across calls (span-level extractions have indices 0..N-1, chunk-level N..M).
- **AC v3-11** Idempotency: second invocation without `--overwrite` writes nothing and makes zero `lx.extract` calls.
- **AC v3-12** Mode-3 guard still rejects `recursive` and `logical`.
- **AC v3-13** `docs/qa-generation.md` Step 2 has 21 span-level + 3 chunk-level class rows; mixed-shape schema example present; configuration block reflects v3 defaults.

Spot-checks:
- At least one `chunk_summary` with non-empty `attributes.summary` and a meaningful `attributes.document_function`.
- At least one `chunk_topic` with `topic_role="primary"` and at least one with `topic_role="secondary"`.
- At least one `pavement_engineering_term` with `attributes.normalized_term` and `attributes.term_category` populated.

## Decisions flagged

- **Two prompts, not one combined.** Span-level extraction is a token-by-token clause-lifting task with 21 classes; chunk-level is a step-back-and-characterize task with 3 classes and a different attribute schema. Combining risks (a) model emitting >1 `chunk_summary` because it sees them as just another class, (b) confusing `chunk_topic` with span-level `finding` / `recommendation`, (c) crowded prompt and unfocused examples. Cost penalty: +1 model call per chunk. Negligible at gpt-4o-mini.
- **`chunk_summary` span = last sentence (`[-1]`).** Predictable, deterministic, demonstrable. Alternative span conventions (first sentence, any representative span) were considered and rejected: arbitrary spans yield inconsistent alignment; first sentence often a markdown header.
- **Description / significance bypass for chunk-level entries.** Those keys are span-level concepts (description = artifact rephrase; significance = source-stated importance). Chunk-level types carry their own type-specific attributes (`summary`, `topic`, `term`, etc.) and don't need the v2 promotion. Bypassing keeps the chunk-level shape compact rather than padding with empty defaults.
- **Continuous `artifact_id` counter** across both calls. Preserves the v2 id format and regex; alternative (per-call namespaced ids) was rejected as a gratuitous break.
- **Sequential calls (not parallel).** Threading is fine but defers cleanly. Wall time doubles per chunk; corpus throughput halves. Fine for fixtures.
- **No code-side quantity enforcement.** Prompt + few-shot is the only steering. Soft validators land if drift is empirically painful.
- **Old prompt files kept on disk.** v1 (`nemo_logic-artifacts.txt`), v2 (`nemo_logic-artifacts-02.txt`) untouched. Deletion deferred until v3 validated on a real corpus.
- **No schema-version marker at the doc level.** v2 SRS OQ-5 still deferred. Per-extraction shape (presence/absence of top-level `description` / `significance`) plus `extractions` bucket keys disambiguate v2 vs v3 in mixed directories.
