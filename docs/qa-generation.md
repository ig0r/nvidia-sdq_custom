# QA Generation

The pipeline has two question-generation flows. They share `path2chunks` (Stage 0.1) but diverge after that:

| Entry point | Bundling | Required mode | Status |
|---|---|---|---|
| `python _nemo.py --sdg` | Multiple chunks per bundle, packed by `RecursiveChunker` under `[llm].max_input_tokens` and de-overlapped by `_trim_overlap_for_context` | `recursive`, `logical`, or `random_logical` | Full pipeline (artifacts → QA pairs → eval) |
| `python _nemo.py --sdg-logical` | One **logical chunk** per bundle (1:1) | `random_logical` only | **Step 1**: writes `-logic-ctx.json` |
| `python extract_artifacts.py` | Per-chunk extraction via Google's `langextract` + OpenAI `gpt-4o-mini` | `random_logical` only | **Step 2** (standalone): writes `-logic-artifacts.json` |

Both flows share `{output_dir}/doc-chunks_{chunk_size}_{method}/` and use disjoint filename suffixes. Future Steps 3 and 4 (`generate_qa_logical.py`, `eval_qa_logical.py`) will follow the same standalone-script pattern.

`--sdg` + `random_logical` continues to work mechanically (logical chunks get re-bundled by `RecursiveChunker`, multi-hop QA runs across the bundles). It is **supported but discouraged** for mode 3 — use `--sdg-logical` + `extract_artifacts.py` instead to get the per-chunk framing those scripts are designed for.

---

## `--sdg-logical` (logical-chunk flow, Step 1)

Generates `{doc_id}-logic-ctx.json` per document — one bundle per logical chunk — and stops there.

### Prerequisite
Set `[chunking].method = "random_logical"` in `cfg/nemo.toml`. With `"recursive"` or `"logical"` the command raises `ValueError` immediately, before any docs are touched. Mode-2 (`logical`) users should run `--sdg` instead.

### Invocations

```bash
# Step 1 only — bundle every logical chunk into -logic-ctx.json, then stop
.venv/bin/python _nemo.py --sdg-logical --cfg cfg/nemo.toml

# Override input/output dirs (same as the other tasks)
.venv/bin/python _nemo.py --sdg-logical \
  --cfg cfg/nemo.toml \
  --input_dir ./rawdata/parsed-techbriefs \
  --output_dir ./data/nemo_briefs_20260422

# Combine with chunking — useful for a fresh corpus where chunks aren't cached.
.venv/bin/python _nemo.py --chunk-only --sdg-logical --cfg cfg/nemo.toml
```

### Output files

```
{output_dir}/doc-chunks_{chunk_size}_random_logical/
  {doc_id}-chunks.json          ← recursive intermediate (path2chunks)
  {doc_id}-logic-chunks.json    ← final logical chunks (path2chunks)
  {doc_id}-logic-ctx.json       ← written by --sdg-logical
  {doc_id}-logic-artifacts.json ← written by extract_artifacts.py (Step 2)
```

`-logic-ctx.json` wraps the entries with `doc_id` so the file self-identifies its source document:
```json
{
  "doc_id": "ABC_123",
  "contexts": [
    {"chunks": [{"chunk_id": 0, "text": "...", "tokens": 123}], "tokens": 123},
    {"chunks": [{"chunk_id": 1, "text": "...", "tokens": 456}], "tokens": 456}
  ]
}
```
Single-element `chunks` list per entry, `tokens` mirrors the chunk's own count. Each chunk additionally carries `source_chunk_ids: list[int]` (passed through from `-logic-chunks.json`). This diverges from the existing `-ctx.json` (a bare JSON array); the standalone Step 2 (`-logic-artifacts.json`) follows the same wrapped-with-`doc_id` convention.

### Re-running / idempotency

`_build_logical_contexts` skip-writes when `Path(out_path).exists() and not self.overwrite`. To force regeneration:
- Delete the `-logic-ctx.json` files, **or**
- Set `self.overwrite = True` in `QAGenerator.__init__` (`_nemo.py:117`).

There is no CLI flag for overwrite.

### Log signals (`CHUNK` level)

```
... | CHUNK | _build_logical_contexts - <file>.md: cache hit -> .../-logic-ctx.json
... | CHUNK | run_sgd_logical_pipeline - <file>.md: N logical-context entries -> .../-logic-ctx.json
```
Plus one warning per oversized bundle:
```
... | CHUNK | _build_logical_contexts - <file>.md: chunk_id=K tokens=T > max_input_tokens=B
```
The warning is informational — Step 1 passes oversized bundles through.

---

## `extract_artifacts.py` (logical-chunk flow, Step 2 — standalone)

Standalone script that reads each `{doc_id}-logic-chunks.json`, sends every logical chunk through Google's [`langextract`](https://github.com/google/langextract) library backed by OpenAI `gpt-4o-mini`, and writes `{doc_id}-logic-artifacts.json` next to it. Lives outside `_nemo.py` because Steps 2–4 of the mode-3 flow have shapes that genuinely differ from the bundled `--sdg` pipeline (per-chunk extraction, single-segment QA, no multi-hop framing).

### Prerequisites

- `[chunking].method = "random_logical"` (the script exits with a clear error otherwise).
- `OPENAI_API_KEY` set in `.env` or the environment (the script fails fast at startup if missing).
- `langextract` installed (`pip install -r reqs.txt`; pinned to `langextract==1.2.1`).
- `*-logic-chunks.json` files already present under `{output_dir}/doc-chunks_{chunk_size}_random_logical/` — typically produced by `--chunk-only` or `--sdg-logical` first.

### Invocations

```bash
# Run after Step 1 (defaults to ./extract_artifacts.toml)
.venv/bin/python extract_artifacts.py

# Point at a different chunk directory (overrides [paths].input_dir)
.venv/bin/python extract_artifacts.py \
  --input_dir ./data/nemo_briefs_20260422/doc-chunks_256_random_logical

# Force regeneration
.venv/bin/python extract_artifacts.py --overwrite

# Backward-compat: read [general].output_dir from the project-wide config and
# auto-discover the single doc-chunks_*_random_logical/ directory under it
.venv/bin/python extract_artifacts.py --cfg cfg/nemo.toml
```

### Extraction-class vocabulary (24 classes, frozen)

Two named subsets, both extracted per chunk via separate `lx.extract` calls:

#### Span-level vocabulary (21 classes)

Normative/functional taxonomy: the classes describe what specific clauses *do* (impose a rule, state a condition, report a finding) rather than the domain entity they name.

| Class | What it captures |
|---|---|
| `requirement` | mandatory or prohibited action — `shall`, `must`, `required`, `prohibited`, or equivalent language |
| `condition` | if/when clause that controls applicability, eligibility, or triggering of another artifact |
| `exception` | case where a normal rule does not apply, or where approval/waiver/special circumstance changes the rule |
| `constraint` | limitation, boundary, restriction, or scope limit |
| `procedure` | ordered steps or workflow for completing a task |
| `method` | analytical, design, testing, calculation, or evaluation approach |
| `formula` | mathematical expression or calculation rule |
| `parameter` | named variable, coefficient, input, output, design value, or material property |
| `threshold` | numeric or categorical boundary used for classification, selection, or decision-making |
| `definition` | explanation of the meaning of a term |
| `actor_role` | person, office, organization, or role responsible for action, review, approval, or consultation |
| `deliverable` | report, form, drawing, submission, record, package, analysis, or other output |
| `assumption` | design premise or condition accepted as true for analysis |
| `finding` | stated observation, result, diagnosis, or conclusion |
| `recommendation` | suggested or preferred action that is not strictly mandatory |
| `best_practice` | preferred, accepted, or commonly recommended practice for design/construction/testing/evaluation/documentation/maintenance |
| `decision` | selected option, approval, rejection, determination, or adopted conclusion |
| `rationale` | explanation of why a requirement, recommendation, decision, finding, or method exists |
| `issue` | identified problem, deficiency, conflict, or gap |
| `risk` | possible adverse outcome, uncertainty, or failure mode |
| `evidence` | data, observation, test result, measurement, or cited basis supporting a finding/decision/rationale/requirement |

#### Chunk-level vocabulary (3 classes)

Compact chunk-wide signals for downstream retrieval, filtering, and entity linking. Extracted via a separate `lx.extract` call against `prompts/nemo_logic-artifacts-03-chunk.txt`.

| Class | What it captures |
|---|---|
| `chunk_summary` | concise source-grounded overview of the whole chunk; `extraction_text` is the chunk's last complete sentence; the actual summary lives in `attributes.summary` (1-2 sentences); also carries `attributes.document_function` and `attributes.scope` |
| `chunk_topic` | normalized topic label for a major subject of the chunk; `attributes.topic` (label) + `attributes.topic_role` (`primary` or `secondary`) |
| `pavement_engineering_term` | important domain term explicitly present in the chunk; `attributes.term` (verbatim), `attributes.normalized_term` (canonical), `attributes.term_category` (e.g. `traffic`, `material`, `test_method`, `design_parameter`) |

**Quantity rules** (prompt-only, not code-enforced): exactly 1 `chunk_summary` per chunk; 1-5 `chunk_topic` per chunk; 0+ `pavement_engineering_term` per chunk (only important domain terms — generic words like "pavement", "design", "project" are excluded unless part of a meaningful technical term).

Out-of-scope classes returned by either call are dropped with an `"NLP"` log line so prompt drift is observable. Each call has its own gate (span call → `SPAN_LEVEL_CLASSES`; chunk call → `CHUNK_LEVEL_CLASSES`); cross-mode emissions are dropped at the per-call gate. The class lists live in `extract_artifacts.py::SPAN_LEVEL_CLASSES`, `::CHUNK_LEVEL_CLASSES`, and the union `::EXTRACTION_CLASSES`.

Do-not-extract types (`table`, `figure`, `reference`, `metadata`, `section_title`, `page_number`, `header`, `footer`, `table_of_contents_entry`, `caption_alone`, `revision_date`, `document_title`) appear only inside span-level `attributes.context_reference` (or `attributes.source_reference`) of a meaningful artifact — never as standalone extractions, never as `pavement_engineering_term` outputs.

### Output schema

`-logic-artifacts.json` wraps a per-chunk artifacts list with the document id. Each extraction carries an `artifact_id` of the form `{doc_id}_chunk_{chunk_id}_art_{idx}` (where `idx` increments across all classes within the chunk), making provenance fully traceable from any single ID.

**Per-extraction shape varies by class group.** Span-level entries have 6 top-level keys (including `description` and `significance`); chunk-level entries have 4 (no `description`/`significance` — those are span-level concepts; the chunk-level canonical content lives inside `attributes`).

```json
{
  "doc_id": "TBF000027_UKN000",
  "artifacts": [
    {
      "chunk_id": 0,
      "tokens": 457,
      "extractions": {
        "requirement": [
          {
            "artifact_id": "TBF000027_UKN000_chunk_0_art_0",
            "text": "separate pavement design analyses shall be prepared",
            "description": "requirement to prepare separate pavement design analyses",
            "significance": null,
            "char_interval": {"start_pos": 145, "end_pos": 197},
            "attributes": {
              "modality": "shall",
              "required_action": "prepared",
              "target": "separate pavement design analyses",
              "source_cue": "shall"
            }
          }
        ],
        "condition": [...],
        "best_practice": [...],
        "chunk_summary": [
          {
            "artifact_id": "TBF000027_UKN000_chunk_0_art_8",
            "text": "Good pavement design practice is to provide adequate drainage to prevent water from being trapped within the pavement structure, which can reduce pavement performance.",
            "char_interval": {"start_pos": 233, "end_pos": 401},
            "attributes": {
              "summary": "The chunk establishes when separate pavement design analyses are required, identifies the designer's responsibility to submit them for District approval, and recommends adequate drainage as best practice.",
              "document_function": "requirement, approval workflow, and best practice",
              "scope": "pavement design analyses for adjacent construction sections"
            }
          }
        ],
        "chunk_topic": [
          {
            "artifact_id": "TBF000027_UKN000_chunk_0_art_9",
            "text": "pavement design analyses",
            "char_interval": {"start_pos": 99, "end_pos": 123},
            "attributes": {"topic": "pavement design analysis", "topic_role": "primary"}
          },
          {
            "artifact_id": "TBF000027_UKN000_chunk_0_art_10",
            "text": "18-kip ESALs",
            "char_interval": {"start_pos": 19, "end_pos": 31},
            "attributes": {"topic": "traffic loading", "topic_role": "secondary"}
          }
        ],
        "pavement_engineering_term": [
          {
            "artifact_id": "TBF000027_UKN000_chunk_0_art_12",
            "text": "18-kip ESALs",
            "char_interval": {"start_pos": 19, "end_pos": 31},
            "attributes": {
              "term": "18-kip ESALs",
              "normalized_term": "18-kip ESAL",
              "term_category": "traffic"
            }
          }
        ]
      }
    },
    {
      "chunk_id": 1,
      "tokens": 198,
      "extractions": {},
      "error": "OpenAI rate limit exceeded"
    }
  ]
}
```

`extractions` keys are a subset of the 24-class vocabulary (omitted when empty). Per-extraction keys depend on the class group:
- **Span-level** (`requirement`, `condition`, …, `evidence`): `{artifact_id, text, description, significance, char_interval, attributes}`. `description` is always a string (defaults to `""` if the model omits it); `significance` is `null` or a non-empty string (the prompt instructs the model to populate it only when the source states or directly supports the artifact's purpose, effect, consequence, or importance); `attributes` carries only type-specific keys (e.g. `modality`, `symbol`, `purpose`) plus the remaining common attributes (`subject`, `scope`, `context_reference`, `source_cue`) — it does *not* carry `description` or `significance`.
- **Chunk-level** (`chunk_summary`, `chunk_topic`, `pavement_engineering_term`): `{artifact_id, text, char_interval, attributes}`. No `description` / `significance` at top level. `attributes` carries the type-specific keys (`summary` / `document_function` / `scope` for `chunk_summary`; `topic` / `topic_role` for `chunk_topic`; `term` / `normalized_term` / `term_category` for `pavement_engineering_term`).

`artifact_id`'s `idx` counter is **continuous across both calls within a chunk** — span-level extractions take the lower indices, chunk-level the higher. `char_interval` is preserved verbatim from `langextract`. Failed chunks emit `"extractions": {}` with an `"error"` field instead of aborting the doc; one error covers both calls.

This **departs from the bundled flow's `-artifacts.json`** in two ways: (a) wrapper object with `doc_id`, (b) per-extraction shape varies by class group versus the bundled flow's flat `text`/`description`/`importance`. The bundled flow's `-artifacts.json` is intentionally not retrofitted. Inside the standalone flow this is also a v1→v2→v3 progression: v1 had `description`/`significance` inside `attributes`; v2 promoted them to top level on every entry; v3 keeps them at top level for span-level entries but omits them entirely for chunk-level entries. Operators with cached v1 / v2 `-logic-artifacts.json` files should re-run with `--overwrite`.

### Configuration

The script's default config is `./extract_artifacts.toml` (dedicated, repo-root). Shape:
```toml
[paths]
input_dir = "./data/nemo_briefs_20260422/doc-chunks_256_random_logical"

[langextract]
model = "gpt-4o-mini"
temperature = 0.0
extraction_passes = 2                          # span-level recall passes
chunk_extraction_passes = 1                    # chunk-level passes (deterministic)
max_char_buffer = 10000                        # disables internal sub-chunking for short logical chunks
prompt_name_span = "nemo_logic-artifacts-03-span"
prompt_name_chunk = "nemo_logic-artifacts-03-chunk"
prompt_lib = "./prompts"
# api_key resolved from .env::OPENAI_API_KEY at runtime if absent here
```

`input_dir` points directly at the chunk directory — outputs (`-logic-artifacts.json`) land in the same directory. Passing `--cfg cfg/nemo.toml` works too: the script reads `[general].output_dir`, globs `doc-chunks_*_random_logical/` under it, and uses the single match (errors if zero or multiple). Validates `[chunking].method == "random_logical"` if present. Missing `[langextract]` keys fall back to `LXConfig` defaults in `extract_artifacts.py`.

### Log signals

```
... | CHUNK | __main__:main - <doc_id>: extracted artifacts from N logical chunks -> .../-logic-artifacts.json
... | CHUNK | __main__:main - <doc_id>: cache hit -> .../-logic-artifacts.json
... | NLP   | __main__:extract - <doc_id> chunk K: dropping out-of-vocab class 'foo'
... | NLP   | __main__:main - <doc_id> chunk K: extraction failed: <message>
```

### Re-running / idempotency

The script skip-writes a doc when `{doc_id}-logic-artifacts.json` exists and `--overwrite` is not passed. To force regeneration: delete the file or pass `--overwrite`.

### Cost note

`langextract` does not surface OpenAI usage uniformly through its API, so cost is currently **uninstrumented**. With `extraction_passes=2` and `chunk_extraction_passes=1`, the script makes **3 model calls per chunk** (2 span + 1 chunk) — roughly 1.5× the v2 per-chunk cost. At gpt-4o-mini input prices, ~$0.0024/chunk. A tiktoken-based post-hoc estimator is a planned follow-up.

---

## `--sdg` (bundled flow, full pipeline)

Runs four LLM stages per document: bundle → artifact extraction → QA generation → LLM-as-judge eval, plus the corpus-wide aggregate. Driven by `_nemo.py::run_sgd_pipeline`; prompts live under `prompts/nemo_*.txt`.

```bash
.venv/bin/python _nemo.py --sdg --cfg cfg/nemo.toml
```

### How a question is built — stage by stage

Once `path2chunks` has produced `{doc_id}-chunks.json`, the SDG pipeline runs three more LLM stages per document. Every stage caches its output and skip-writes when present.

#### Stage 1 — Bundle chunks into LLM-sized contexts (`extract_artifacts`)
The chunk list is handed to `aisa/parse/chunk.py::RecursiveChunker(custom_input=chunks, max_input_tokens=...)` which packs **consecutive chunks** until they would overflow the LLM's input budget. Each pack becomes one **bundle**.

Then `_trim_overlap_for_context` walks each bundle and strips the suffix/prefix overlap between neighboring chunks (the recursive splitter introduced overlap on purpose for retrieval; inside one prompt it's wasted tokens).

The de-overlapped bundles are written to `-ctx.json`. So we now have N bundles, each a list `[{chunk_id, text, tokens}, …]` totaling ≤ `[llm].max_input_tokens`.

#### Stage 2 — Extract "facts" per bundle (`extract_artifacts`, same call)
Each bundle's joined text is sent through `nemo_artifacts`. The LLM returns up to `max_artifacts` items in 8 buckets: `key_concepts`, `relationships`, `themes`, `entities`, `processes`, `insights`, `technical_terms`, `contextual_factors`. Output → `-artifacts.json`.

`get_fact_blocks` formats each bundle's artifacts into an XML-ish `<key_concepts>… </key_concepts><relationships>… </relationships>…` string — one **`facts_block`** per bundle.

`get_ctx_blocks` reads `-ctx.json` and re-formats each bundle as `=== Section 1 ===\nSegment {chunk_id}: {text}\n…` — one **`context_block`** per bundle.

So for each bundle we now have a paired `(facts_block, context_block)`.

#### Stage 3 — Generate QA pairs (`generate_qa_pairs`)
For every bundle, one call to `nemo_qa-gen` is made with:

- `facts_block` — *what to ask about* (the artifacts).
- `context_block` — *what evidence the LLM can draw from* (the actual segments, with their `chunk_id`s).
- Hardcoded counts from `QAGenerator.__init__` (`_nemo.py:118-132`):
  - **Query types** (each question gets one): `multi_hop` (5), `structural` (5), `contextual` (5).
  - **Reasoning types** (orthogonal axis): `factual` / `relational` / `inferential` / `temporal` / `procedural` (3 each), `visual` / `causal` (0).
  - `min_hops=2`, `max_hops=3`, `min_complexity=3`, `num_pairs=15`.

The prompt instructs the model to:
- Ask questions that require **connecting multiple segments** (no shallow lookups).
- Never reference "the transcript" / "the context" — questions must read standalone.
- Tag each question with both a `query_type` AND a `reasoning_type` (orthogonal fields).
- Record `segment_ids` — the chunk IDs that are the **source material** for the question. These become the positives (`pos_doc`) downstream in `--prep`.
- For multi-hop, include `hop_contexts` listing per-hop segment IDs and a summary.

Output → `-qa_pairs.json`: one entry per bundle, each with a `pairs` array of ~15 `{question, answer, query_type, reasoning_type, question_complexity, segment_ids, hop_count, hop_contexts}` records.

To change the counts, edit the constants in `QAGenerator.__init__` — they are not exposed in TOML.

#### Stage 4 — Judge them (`evaluate_qa_pairs`)
Each bundle's QA pairs + the same `context_block` go to `nemo_eval`. The judge scores each pair. Pairs with `overall.score < 7.0` are dropped later in Stage 1 of `--prep` (`filter_and_convert`, threshold configurable via the `quality_threshold` argument).

### Data flow

```
chunks (-chunks.json)
  └─► RecursiveChunker bundling + overlap trim
        ├─► -ctx.json   ── format ──► context_block (segments w/ chunk_ids)
        └─► nemo_artifacts ── -artifacts.json ── format ──► facts_block
              └─► nemo_qa-gen(facts_block, context_block, counts)
                    └─► -qa_pairs.json (questions tagged with segment_ids)
                          └─► nemo_eval ── -qa_eval.json (filter < 7.0 in --prep)
```

The crucial design point: `facts_block` tells the LLM *what's interesting* (distilled), while `context_block` keeps the raw segments with their `chunk_id`s in view so the model can both ground answers and emit accurate `segment_ids` — those IDs are what stitches each question back to its positive chunks downstream.

### Outputs (per doc, in the same chunk dir)
```
{doc_id}-ctx.json        bundles after RecursiveChunker packing + overlap trim
{doc_id}-artifacts.json  per-bundle artifact extraction (nemo_artifacts)
{doc_id}-qa_pairs.json   per-bundle QA pairs (nemo_qa-gen)
{doc_id}-qa_eval.json    per-bundle judge scores (nemo_eval)
{doc_id}-sdg.json        per-doc combined record
```
Plus `{output_dir}/full_sdg_output.json` — the corpus-wide aggregate consumed by `--prep`.
