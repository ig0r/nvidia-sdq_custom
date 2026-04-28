# Plan: Extract-Artifacts v2 — Normative 21-Class Taxonomy

## Context

The current `extract_artifacts.py` extracts 8 pavement-noun classes (`material`, `distress`, `treatment`, `specification`, `test_method`, `metric`, `process`, `reference`). That taxonomy captures *what things appear* but not *what the text does* with them — and "what the text does" (rules, conditions, evidence, recommendations, decisions) is what downstream QA generation needs to ask about.

The new spec replaces the taxonomy with a 21-class **normative/functional** vocabulary (`requirement`, `condition`, `procedure`, `parameter`, `threshold`, `actor_role`, `deliverable`, `finding`, `recommendation`, `evidence`, etc.) plus an explicit *do-not-extract* list (`table`, `figure`, `reference`, headers/footers/page numbers — these only appear inside `attributes.context_reference`). The new prompt enforces source-grounding with a strict `significance` rule (only when stated in source) and clearer boundary rules between overlapping types (e.g. `requirement` vs `recommendation` vs `best_practice`).

The script's plumbing — standalone single file, mode-3 input dir, `doc_id` wrapper, `artifact_id` per extraction, idempotency, failure isolation — stays exactly the same. Only the **vocabulary, prompt, example, and a few langextract knobs** change.

## Scope

**In scope**
- New prompt file `prompts/nemo_logic-artifacts-02.txt` containing the `prompt_description` text from §9 of the recommendation, verbatim.
- Replace `EXTRACTION_CLASSES` (8 → 21) and `PAVEMENT_EXAMPLES` (current 13-extraction example → the single 6-extraction example from §10) in `extract_artifacts.py`.
- Update `extract_artifacts.toml` `[langextract]`: `extraction_passes 3 → 2`, `prompt_name → "nemo_logic-artifacts-02"`. (`max_char_buffer` stays at 10000.)
- **Schema change**: promote `description` and `significance` to top-level per-extraction keys; remove them from `attributes`. Per-extraction shape becomes `{artifact_id, text, description, significance, char_interval, attributes}`. `description` is always present (default `""` if model omits); `significance` is `null` when the source does not state it.
- Update `docs/qa-generation.md` Step 2 section: new class table, schema example with one of the new classes and the new top-level keys, configuration block reflects new defaults.

**Out of scope**
- Provider switch. **Confirmed**: keep OpenAI `gpt-4o-mini`, `OPENAI_API_KEY` resolution.
- Doc-level wrapper (`{doc_id, artifacts: [{chunk_id, tokens, extractions, [error]}]}`) and `artifact_id` format — unchanged.
- Mode-3 guard, idempotency, failure isolation — unchanged.
- `lx.extract` call signature — unchanged from v1 (no new kwargs; `max_char_buffer` stays at 10000, no `max_workers`).
- Few-shot example **input format** — langextract's `lx.data.Extraction` requires `description`/`significance` inside `attributes`. Only the **output transformation** changes: the script pops both out of langextract's attributes dict and elevates them to top-level keys on the emitted entry.
- Type-specific attribute enforcement in code. The new spec defines per-class attribute schemas (e.g. `requirement.modality`, `parameter.symbol`) — we leave attribute production to prompt + few-shot steering; the script only enforces class-name vocabulary, mirroring current behavior.
- Stages 2–4 of the broader pipeline (artifact filter, QA gen, QA validation) — separate work.
- Deleting the old `prompts/nemo_logic-artifacts.txt`. Kept on disk for reference until v2 is validated; deletion is a follow-up.
- Bumping few-shot coverage beyond the single provided example (5 of 21 classes covered: `condition`, `requirement`, `actor_role`, `deliverable`, `best_practice`). Supplementing with manual- and report-style examples is a deliberate follow-up.

## Concrete changes

### New: `prompts/nemo_logic-artifacts-02.txt`
Verbatim copy of the `prompt_description` from §9 of the spec — the full block including the 21-class definitions, common-attribute glossary, boundary rules, and do-not-extract list. No `{placeholders}`.

### Modify: `extract_artifacts.py`

**`EXTRACTION_CLASSES`** — replace the 8 strings with:
```python
EXTRACTION_CLASSES: list[str] = [
    "requirement", "condition", "exception", "constraint",
    "procedure", "method", "formula", "parameter",
    "threshold", "definition", "actor_role", "deliverable",
    "assumption", "finding", "recommendation", "best_practice",
    "decision", "rationale", "issue", "risk", "evidence",
]
```

**`PAVEMENT_EXAMPLES`** — replace the current 13-extraction example with the single 6-extraction example from §10 of the spec (text about ESALs/design analyses/drainage; extractions for `condition`, `requirement`, `actor_role`, `requirement` (compound), `deliverable`, `best_practice`). The example data keeps `description` and `significance` inside each `Extraction.attributes` (langextract's required input shape).

**`LXConfig`** — change two defaults: `extraction_passes: int = 2` (was 3) and `prompt_name: str = "nemo_logic-artifacts-02"` (was `"nemo_logic-artifacts"`). `max_char_buffer` stays at 10000. No new fields.

**`PavementExtractor.extract`** — `lx.extract(...)` call signature unchanged from v1 (no new kwargs). The per-extraction transformation changes: pop `description` and `significance` out of `ext.attributes` and elevate them to top-level keys on the emitted entry:
```python
attrs = dict(ext.attributes or {})
description = attrs.pop("description", "")
significance = attrs.pop("significance", None)
entry = {
    "artifact_id": f"{doc_id}_chunk_{chunk_id}_art_{ext_idx}",
    "text": ext.extraction_text,
    "description": description,
    "significance": significance,           # may be None
    "char_interval": char_iv,
    "attributes": attrs,                    # type-specific + remaining common attrs only
}
```

The class-vocabulary drop logic (`if cls not in EXTRACTION_CLASSES: ... continue`) keeps working unchanged — it now filters against the 21-class set. The do-not-extract names (`table`, `figure`, `reference`, etc.) will be dropped at this gate if the model emits them despite the prompt instruction; an `"NLP"` log line per drop surfaces prompt drift.

### Modify: `extract_artifacts.toml`
```toml
[langextract]
model = "gpt-4o-mini"                     # unchanged
temperature = 0.0                          # unchanged
extraction_passes = 2                      # was 3
max_char_buffer = 10000                    # unchanged — keeps full-chunk context for the LLM call
prompt_name = "nemo_logic-artifacts-02"    # was "nemo_logic-artifacts"
prompt_lib = "./prompts"                   # unchanged
```

### Modify: `docs/qa-generation.md`
- Replace the 8-row class table in the Step 2 section with the 21-row class table (one short definition per class, mirroring §3 of the spec).
- Update the schema JSON example to use one of the new classes (e.g. `requirement` or `parameter` instead of `material`).
- Update the configuration code block to match the new `[langextract]` defaults.
- Add a one-line note: do-not-extract types (`table`, `figure`, `reference`, etc.) appear only inside `attributes.context_reference` per the new prompt, not as standalone extractions.

### Unchanged
- `extract_artifacts.py` plumbing: argparse, `_resolve_input_dir`, idempotency, failure isolation, `artifact_id` minting, output writing.
- `cfg/nemo.toml` (project-wide config) — backward compat with `--cfg cfg/nemo.toml` continues to work via the input-dir glob.
- `_nemo.py:439` mode-3 guard.
- `reqs.txt` — no new dependencies.

## Behavior notes

- **`max_char_buffer = 10000` (unchanged from v1)**: keeps full-chunk context in the LLM call. Mode-3 chunks (typical 1000–2000 chars; max ~8000 chars at `hybrid_window=8 × chunk_size=256` tokens × ~4 chars/token) all fit in one call. The recommendation's 1500 was rejected as ill-suited to pre-chunked input — sub-chunking would undo upstream coherence and weaken the new normative-taxonomy boundary cues that benefit from cross-sentence context.
- **`extraction_passes = 2`**: 33% cost reduction vs. the current 3.
- **Schema break vs. v1** (per-extraction object): each entry now has top-level `text`, `description`, and `significance` alongside `artifact_id`, `char_interval`, and `attributes`. `attributes` no longer carries `description` or `significance` — only type-specific keys (e.g. `modality`, `symbol`, `purpose`) and the remaining common attributes (`subject`, `scope`, `context_reference`, `source_cue`). Consumers who looked up `attributes.description` or `attributes.significance` in v1 outputs need to read the top level for v2.
- **Doc-level shape unchanged**: top-level `{doc_id, artifacts: [...]}` and per-chunk `{chunk_id, tokens, extractions, [error]}` remain identical to v1.
- **Cache invalidation**: existing `-logic-artifacts.json` files were produced under the v1 vocabulary and the v1 per-extraction shape. They're still valid JSON and the script will skip-write them. To regenerate under v2, delete them or run with `--overwrite`.

## Verification

Re-run the existing TBF000027 smoke fixture:

```bash
mkdir -p /tmp/extract_smoke/doc-chunks_256_random_logical
cp data/techbriefs_20260427/doc-chunks_256_random_logical/TBF000027_UKN000-logic-chunks.json \
   /tmp/extract_smoke/doc-chunks_256_random_logical/

.venv/bin/python extract_artifacts.py \
  --input_dir /tmp/extract_smoke/doc-chunks_256_random_logical
```

Re-run the AC harness from prior smoke tests:
- **AC-1** doc_id wrapper + count parity → unchanged
- **AC-2** chunk_id + tokens parity → unchanged
- **AC-3** char_interval validity → unchanged
- **AC-4** artifact_id format `^.+_chunk_\d+_art_\d+$` + uniqueness → unchanged
- **AC-5 (new)** every emitted class ∈ the 21-class set; no `material` / `distress` / etc. anywhere; no `table` / `figure` / `reference` either
- **AC-6 (new)** every emitted entry has top-level keys `{artifact_id, text, description, significance, char_interval, attributes}`. `attributes` contains no `description` or `significance` key. `description` is a string (possibly `""`); `significance` is `null` or a non-empty string.
- **AC-8** idempotency → unchanged

Spot-check a few extractions:
- At least one `requirement` (with `attributes.modality` like "shall"/"must")
- At least one `parameter`, `procedure`, or `method`
- `attributes.context_reference` populated where the source mentions a table/figure/section inside a meaningful artifact
- Top-level `significance` non-null only where the source explicitly states why something matters

## Decisions flagged

- **Provider** — keeping OpenAI `gpt-4o-mini` (user-confirmed). The recommendation defaults to Gemini 2.5 Flash; not adopted to keep `OPENAI_API_KEY` as the only key required and stay consistent with the project's chat LLM choice.
- **`max_char_buffer` stays at 10000** — the recommendation's 1500 is calibrated for raw documents where you want langextract to sub-chunk for recall. Our mode-3 chunks are pre-sized to be coherent semantic units (~1000–2000 chars typical, ~8000 chars max). Sub-chunking would undo upstream coherence. With sub-chunking off, `max_workers` is dropped as a no-op.
- **Schema break is intentional and surfaced** — promoting `description`/`significance` to top-level is a v1→v2 incompatibility, but v2 is already incompatible at the class-vocabulary level (a v1 consumer expecting `material` keys will fail on v2 output regardless). The break is therefore "free" — clean rather than smuggled. Operators must regenerate cached files via `--overwrite`.
- **`description` defaults to `""`, `significance` defaults to `None`** — the prompt asks for both, but the model may omit. Defaulting keeps the schema regular for downstream consumers; a missing-description rate becomes observable via post-hoc inspection.
- **Old prompt file** — `prompts/nemo_logic-artifacts.txt` stays on disk after this change. Once v2 is validated on a real corpus, deletion is a clean follow-up. The `-02` suffix on the new file makes the version split explicit.
- **Type-specific attributes** — not enforced in code. The prompt + few-shot example are the steering mechanism. If we see attribute drift in real runs, we can add a post-hoc validator in a follow-up (similar to the current class-vocab drop).
- **Few-shot coverage** — single 6-extraction example covering 5 of 21 classes (`condition`, `requirement`, `actor_role`, `deliverable`, `best_practice`). Adequate for the manual/specification document type that dominates the corpus; supplementing with a report-style example to cover `finding`/`evidence`/`issue`/`risk`/`decision` is a deliberate follow-up once we see baseline output quality.
