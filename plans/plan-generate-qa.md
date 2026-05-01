# Plan: Question Generation on Logical-Chunk Artifacts (Step 3, Standalone)

## Architectural framing

Step 3 of the mode-3 (`random_logical`) logical-chunk SDG flow. Steps 1 and 2 are described in `plans/plan-sdg-logical.md` and `plans/plan-sdg-logical-step2.md`. Step 1 (`--sdg-logical`) writes `-logic-ctx.json` (one bundle per logical chunk). Step 2 (`extract_artifacts.py`) writes `-logic-artifacts.json` (per-chunk span-level extractions + chunk-level signals via `langextract` + OpenAI Structured Outputs). Step 3 (this plan, `generate-qa.py`) consumes both files and writes `generated-questions.json` + `.csv` via a two-phase async pipeline:

1. **Phase 1 — QA generation.** For each `(context, artifact)` pair, one LLM call produces 3-5 questions.
2. **Phase 2 — Citation extraction.** For each generated question, one LLM call extracts a verbatim citation from the surrounding context.

Step 3 is the second of three planned standalone scripts (the first being `extract_artifacts.py`); Step 4 (`eval_qa_logical.py`, an LLM-as-judge) remains deferred. The asymmetry — Step 1 in-pipeline, Steps 2-4 standalone — is deliberate: per-chunk framing, single-segment QA without `multi_hop`/`hop_contexts` machinery, and tunable concurrency are awkward to bolt onto `_nemo.py::run_sgd_pipeline` (the bundled `--sdg` flow), which assumes multi-segment bundles.

`--sdg` + `random_logical` continues to work mechanically (logical chunks get re-bundled by `RecursiveChunker`, multi-hop QA runs across the bundles). It is **supported but discouraged** for mode 3 — use the standalone Step 2 + Step 3 scripts to get the per-chunk framing this plan is designed for.

## Heritage

`generate-qa.py` is adapted from the working two-phase generator in `examples/qa-generation/generate-data-async2.py`. The example's structure (two phases, async with semaphore, retry loop, periodic save, intermediate-file resume, structured outputs per provider) is preserved. The inputs are remapped onto our pipeline's data model:

| `generate-data-async2.py` (process-citation flow) | `generate-qa.py` (logical-chunk flow) | Source |
|---|---|---|
| `process` item | one **context** entry (one `u_ctx_id`) | `*-logic-ctx.json::contexts[i]` |
| `process.citation.citation` (`{SOURCE_TEXT}`) | concatenated chunk texts of that context (`{CONTEXT}`) | `*-logic-ctx.json::contexts[i].chunks[].text` |
| `process.source_metadata` (`{DOCUMENT_INFO}`) | `doc_id` + `chunk_signals.summary` (summary, scope, document_functions) + topics | `-logic-artifacts.json::artifacts[i].chunk_signals` |
| `elements` (input/output/step/general) | flattened **artifacts** across configured categories + a synthetic `summary` element drawn from `chunk_signals.summary` | `-logic-artifacts.json::artifacts[i].extractions` + `chunk_signals` |
| `process_id`, `source_chunk_id` | `doc_id`, `u_ctx_id`, `u_logic_chunk_id`, `source_u_chunk_ids` | both files |
| `chapters` filter | dropped — no chapter info in tech briefs | — |
| missing-citation filter | dropped — context text is the citation source itself | — |

Iteration unit: one `(context, artifact)` task → one LLM call → 3-5 questions. `question_element_type` becomes the artifact category (`finding`, `procedure`, …, plus `summary` for the synthetic element).

## Scope

In scope:
- Mode-3 only inputs (`*-logic-ctx.json` + `*-logic-artifacts.json`); the script discovers them by globbing a single chunk directory.
- Two-phase pipeline (Phase 1 QA gen → Phase 2 citation extract) with independent concurrency knobs.
- One LLM call per `(ctx, artifact)` pair in Phase 1; one LLM call per question in Phase 2.
- Three providers: OpenAI (`gpt-*`), Google Gemini (`gemini-*`), Ollama (any other model name) — copied from the example, less Mistral / Claude / DeepSeek paths.
- Pavement-engineering framing preserved in both prompts.
- Resume from prior intermediate files; periodic save during each phase.
- Final JSON + CSV outputs with deterministic, source-ordered `question_id`s.

Out of scope:
- Mistral, Claude, DeepSeek client paths.
- Chapter filtering or `source_metadata` `ast.literal_eval` parsing.
- Pushover notifications.
- Multi-chunk contexts: structurally supported (the code joins `chunks[].text`) but Step 2 currently emits one chunk per ctx, so it is untested at >1.
- Integration into `_nemo.py`'s `tasks` dict — this is a standalone driver.
- LLM-as-judge evaluation (deferred to Step 4).

## Design choices

1. **Self-contained script.** No imports from `examples/`. Helpers (`read_configuration`, `load_json`, `save_to_json`, `replace_symbols`, `load_prompt_from_file`) and Pydantic models (`GeneratedQuestion`, `GeneratedQuestionsResponse`, `CitationResponse`, `QuestionDifficulty`, `QuestionType`) are inlined. The script lives at the repo root alongside `_nemo.py` and `extract_artifacts.py`.

2. **Provider routing identical to the example.** Three discriminators (`is_gpt`, `is_gemini`, `is_ollama_model`) plus three structured-output wrappers:
   - `query_openai_structured` → `client.responses.parse(text_format=PydanticClass)`.
   - `query_gemini_structured` → `genai.GenerativeModel.generate_content(generation_config={"response_mime_type": "application/json", "response_schema": JSON_SCHEMA_DICT})`.
   - `query_ollama_structured` → `client.chat(format=PydanticClass.model_json_schema())`.

3. **Lazy `google.generativeai` import.** The example uses the legacy `google.generativeai` SDK; the project's `reqs.txt` ships `google-genai` (a different package). The legacy import is wrapped in `try/except ImportError`; if a `gemini-*` model is selected and the legacy SDK is missing, `initialize_client` raises a clear `RuntimeError`. The script imports cleanly without it for OpenAI/Ollama operators.

4. **Iteration unit = `(context, artifact)`.** `build_tasks` flattens each context's `extractions{category: [...]}` across the configured `artifact_categories`. `max_artifacts_per_ctx > 0` caps the per-ctx flatten; `0` means no cap. One LLM call per task → 3-5 questions. Artifacts within a category preserve their source order; categories are visited in TOML-listed order.

5. **Synthetic `summary` element.** When `include_summary_element = true`, `build_summary_artifact` constructs a one-off element from `chunk_signals.summary` (text = `summary`, attributes = `{scope, document_functions}`). It is appended after the real artifacts, so its questions land at the end of each ctx's question block. `question_element_type = "summary"` distinguishes them downstream.

6. **Two-phase resume.**
   - **Phase 1** keys on `(u_ctx_id, artifact_id)`. The intermediate file `generated-questions_qa_only.json` is loaded on startup; tasks whose key is already represented are skipped. Each question carries a TEMP- ID (`f"TEMP-{u_ctx_id}-{artifact_id}-{q_idx}"`) until the final renumbering pass.
   - **Phase 2** keys on `question_id`. The intermediate file `generated-questions_with_citations.json` is loaded, citations from prior runs are merged onto the in-memory questions, and only `citation_extracted == false` questions are sent to the LLM.

7. **Final ordering.** Phase 1 uses `asyncio.as_completed`, so questions are appended to `all_questions` in finish-order (non-deterministic). Right before the final renumbering loop, the script sorts by `(doc_id, u_ctx_id, artifact_id)` using a **natural key** (digit-runs compared as integers). This makes `ctx-9` precede `ctx-10` and yields stable, source-ordered output across runs. Final IDs are then assigned per-ctx: `f"{u_ctx_id}-q-{n}"`, with `n` resetting to 0 at each new ctx. The intermediate files keep TEMP- IDs so resume keys remain stable across partial runs.

8. **Periodic save.** Phase 1 saves every `periodic_save_interval_qa` completed tasks (default 10); Phase 2 every `periodic_save_interval_citations` completed questions (default 50). One final save closes each phase.

9. **Embedded schemas, two forms.** The Pydantic classes are used directly with the OpenAI `text_format=` parameter. For Gemini's `response_schema` and Ollama's `format=` parameter, parallel `QUESTIONS_SCHEMA` and `CITATION_SCHEMA` JSON-Schema dicts are defined alongside the Pydantic classes. The classes and dicts must stay in sync; both are 4-key (questions schema) / 3-key (citation schema) shapes.

## Files

- `generate-qa.py` (repo root) — the script.
- `generate-qa.toml` (repo root, alongside the script) — config (chunk_dir, output paths, prompts, models, concurrency, artifact categories, symbol replacement).
- `prompts/nemo_qa-gen-artifact.txt` — QA-gen prompt with placeholders `{CONTEXT}`, `{DOCUMENT_INFO}`, `{ARTIFACT_CATEGORY}`, `{ARTIFACT}`.
- `prompts/nemo_extract-citation.txt` — citation prompt with placeholders `{CONTEXT}`, `{QUESTION}`.

## Prompt placeholders

| Placeholder | Built by | Source data |
|---|---|---|
| `{CONTEXT}` | `build_context_text(ctx_entry)` — joins `chunks[].text` with `\n\n` | `*-logic-ctx.json::contexts[i].chunks[].text` |
| `{DOCUMENT_INFO}` | `build_doc_info(doc_id, artifact_entry)` — formats summary + scope + document_functions + topics | `*-logic-artifacts.json::artifacts[i].chunk_signals` |
| `{ARTIFACT_CATEGORY}` | category key (loop variable) | one of `artifact_categories` from TOML, or `"summary"` for the synthetic element |
| `{ARTIFACT}` | `format_artifact(category, artifact)` — text + description + significance + attributes | one entry from `extractions[category]`, or `build_summary_artifact(artifact_entry)` |
| `{QUESTION}` | the generated `question` field | Phase 1 output |

Symbol replacement (when `replace_symbols = true`) is applied to the context text only — both at QA generation time (so the LLM sees normalized characters) and at citation extraction time (to keep the Phase 2 prompt consistent with what was used in Phase 1).

## Output schema (per question)

```json
{
  "question_id": "{u_ctx_id}-q-{n}",
  "question": "...",
  "answer": "...",
  "full_citation": {"citation": "...", "first_sentence": "...", "last_sentence": "..."},
  "question_type": "factual|conceptual|application|analysis",
  "question_difficulty": "basic|intermediate",
  "question_element_type": "finding|issue|method|procedure|recommendation|best_practice|rationale|summary",
  "doc_id": "TBF000011_UKN000",
  "u_ctx_id": "TBF000011_UKN000-ctx-0",
  "u_logic_chunk_id": "TBF000011_UKN000-logic-chunk-0",
  "source_u_chunk_ids": ["TBF000011_UKN000-chunk-1", "TBF000011_UKN000-chunk-2"],
  "artifact_id": "TBF000011_UKN000_chunk_0_art_0",
  "u_artifact_id": "TBF000011_UKN000-ctx-0-art-0",
  "artifact": { /* full artifact dict (text, description, significance, attributes) */ },
  "context_text": "...",
  "model_qa": "gpt-4o-mini",
  "model_citation": "gpt-4o-mini"
}
```

The CSV output (`generated-questions.csv`) flattens this list into rows, stringifying `source_u_chunk_ids` and `artifact` for spreadsheet legibility; its `citation` column is sourced from `full_citation.citation`. The internal `citation_extracted` flag is dropped from the final JSON/CSV.

**Citation field shape.** The verbatim citation string lives at `full_citation.citation`; there is no top-level `citation` field. `full_citation` also carries `first_sentence` and `last_sentence` (currently unused by any consumer, but preserved for future citation-alignment / validation tooling).

## Output ordering

After both phases complete, `main()` performs:

```python
def _natural_key(s: str) -> tuple:
    import re
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", s or ""))

all_questions.sort(key=lambda q: (
    _natural_key(q.get("doc_id", "")),
    _natural_key(q.get("u_ctx_id", "")),
    _natural_key(q.get("artifact_id", "")),
))
```

Then a per-ctx counter assigns final `question_id`s. Intermediate files (`*_qa_only.json`, `*_with_citations.json`) keep TEMP- IDs verbatim — they are resume-state, not user-facing artifacts.

## Critical files

- `generate-qa.py` (new, repo root) — the script.
- `generate-qa.toml` (new, repo root) — config alongside the script.
- `prompts/nemo_qa-gen-artifact.txt` (new) — QA-gen prompt.
- `prompts/nemo_extract-citation.txt` (new) — citation prompt.
- `examples/qa-generation/generate-data-async2.py` — reference structure (untouched).
- `examples/qa-generation/prompts/{generate-data-04.txt, extract-citation-01.txt}` — base prompts (untouched).

Reused: nothing from `aisa/gen` (the example's raw-client routing is preserved verbatim per the design choice). The script does NOT use `BaseLLM.run_chain` / `Embedder.encode` / the cost decorators — by design.

## Verification

```bash
cd /Users/igor/dev/llm/pavement-gpt/nvidia-pipeline/nvidia-sdq_custom

# Smoke run on the test fixture (default config)
.venv/bin/python generate-qa.py --config generate-qa.toml

# Resume sanity: re-run; both phases should skip-write.
.venv/bin/python generate-qa.py --config generate-qa.toml

# Phase-2-only re-run: delete the citations file, re-run; Phase 1 picks up from
# qa_only.json (no LLM calls), Phase 2 re-extracts citations.
rm data/_test/qa-gen/generated-questions_with_citations.json
.venv/bin/python generate-qa.py --config generate-qa.toml

# Output assertions
.venv/bin/python -c "
import json, re
def nk(s): return tuple(int(p) if p.isdigit() else p for p in re.split(r'(\d+)', s or ''))
data = json.load(open('data/_test/qa-gen/generated-questions.json'))
keys = [(nk(q['doc_id']), nk(q['u_ctx_id']), nk(q['artifact_id'])) for q in data]
assert keys == sorted(keys), 'natural-sort order broken'
# question_id sequential per ctx
from collections import defaultdict
seen = defaultdict(list)
for q in data:
    seen[q['u_ctx_id']].append(int(q['question_id'].rsplit('-q-', 1)[1]))
for ctx, ids in seen.items():
    assert ids == list(range(len(ids))), f'gap in {ctx}: {ids}'
print(f'{len(data)} questions, {len(seen)} contexts — order OK')
"

# Provider switch (Gemini, requires legacy google-generativeai installed)
sed -i.bak 's/^model_qa = .*/model_qa = "gemini-2.5-flash"/' generate-qa.toml
.venv/bin/python generate-qa.py --config generate-qa.toml
mv generate-qa.toml.bak generate-qa.toml

# Provider switch (Ollama, requires the model installed locally)
sed -i.bak 's/^model_qa = .*/model_qa = "qwen3:4b"/' generate-qa.toml
.venv/bin/python generate-qa.py --config generate-qa.toml --host localhost --port 11434
mv generate-qa.toml.bak generate-qa.toml
```

Smoke run reference (gpt-4o-mini, 2-doc fixture, `max_artifacts_per_ctx = 0`, summary on):
- 33 unique `u_ctx_id` (3 from `TBF000011_UKN000`, 30 from `TBF000131_UKN000`).
- 1313 questions across all 33 ctx (Phase 1).
- All 1313 receive citations (Phase 2 ~75 s at `max_concurrent_citations = 20`).
- `generated-questions.json` and `generated-questions.csv` agree row-for-row.

## Risks / known limitations

- **Verbatim-citation drift.** The Phase 2 prompt asks for an exact substring of `{CONTEXT}`, but the model occasionally paraphrases (~1 in 5 sampled in the smoke run). Downstream consumers should not assume `citation in context_text` is always true.
- **Legacy Gemini SDK requirement.** Gemini support requires `pip install google-generativeai` separately; the project's `reqs.txt` ships `google-genai` (the new SDK), which has a different API.
- **Ordering depends on the natural sort.** Without the sort, `generated-questions.json` reflects Phase 1 finish order — non-deterministic across runs. The sort is the source of truth for ordering; intermediates are intentionally not sorted.
- **No cost telemetry.** The example's raw-client routing bypasses `aisa/gen`'s `ChatResponse` decorator, so token / cost / timing logs are not aggregated. A tiktoken-based post-hoc estimator is a planned follow-up (mirrors the gap noted for `extract_artifacts.py` v4).
- **Multi-chunk per ctx.** Structurally supported (`build_context_text` joins `chunks[].text`) but Step 2 currently emits exactly one chunk per ctx, so this code path is untested.
- **Symbol replacement scope.** Applied to context text only — not to the artifact text or the document info. If a problematic glyph lives only in the artifact, it will reach the model unmodified.
