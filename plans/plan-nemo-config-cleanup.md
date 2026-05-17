# Plan: Conservative config cleanup for `cfg/nemo.toml` + `cfg/nemo_specs.toml`

**Companion SRS:** `plans/srs-nemo-config-cleanup.md` (authoritative — this plan refers to its
sections, does not duplicate them)
**Status:** Implemented
**Target:** delete dead sibling-project carryover from both `_nemo.py` TOML configs; conservative
scope only (no `.py` change, no behavior change to any mode); both files stay multi-mode capable
**Working notes (background, not authoritative):**
`/Users/igor/.claude/plans/perform-research-what-settings-elegant-hartmanis.md`

## Why

`cfg/nemo.toml`/`cfg/nemo_specs.toml` carry ~110 lines of cruft from a larger sibling project
(`[pub242]`, `[jsa]`, `[nlp]`, `[doc]`, `[general.theme]`, `[general.tasks]`,
`[general].metadata_folder`). A completed, this-session-verified audit (SRS §2) shows **zero**
code paths in `_nemo.py`, `extract_artifacts.py`, or `aisa/` read any of it. Removing it shrinks
each file from 143 → ~33 lines with provably no behavior change. Conservative scope: delete only
top-level items no script reads; keep everything multi-mode.

## Pipeline

This is a config-hygiene change; it sits beside (does not alter) the routes:

```
unchanged: _nemo.py --chunk-only/--sdg/--sdg-logical/--prep --cfg cfg/nemo.toml
unchanged: _nemo.py --sdg-logical --cfg cfg/nemo_specs.toml → extract_artifacts.py → generate-qa.py → self-check-qa.py
changed  : cfg/nemo.toml      (trim dead sections + header)
changed  : cfg/nemo_specs.toml (identical structural trim + header)
```

## Decisions (user)

- **Scope = CONSERVATIVE** (explicit user choice): delete only top-level sections/keys NO script
  reads; keep both files multi-mode capable. Moderate/aggressive trims are *not* the chosen
  approach — recorded as deferred future work (SRS §10).
- **Both files edited identically** (same structural deletion; value-level differences
  preserved per SRS FR-2.4).
- `[langextract]` **kept in both**, including its dead-copy presence in `cfg/nemo_specs.toml`
  (parity over minimalism, under conservative scope) — SRS §2 finding, FR-1.3.
- **No `.py` change** — load-bearing safety property (SRS FR-5).

## Change 1 — `cfg/nemo.toml`: delete dead carryover + add header

Per SRS FR-1.1 / FR-3, against the verified current layout (143 lines):
- Add a one-line `#` header at the top recording the file is consumed only by `_nemo.py` (+
  `extract_artifacts.py` for `[langextract]` via `--cfg cfg/nemo.toml`).
- Delete in full: `[general].metadata_folder` (l.5), `[general.theme]` (6–9),
  `[general.tasks]` (10–14), `[pub242]`+`.tasks`+`.saliency` (16–40), `[jsa]`+`.tasks` (87–99),
  `[nlp]`+`.tasks` (101–110), `[doc]`+`.rag`+`.mlp`+`.tasks` (112–143).
- Keep `[general]` (only `output_dir`, `data_dir` + their inline history comments), `[llm]`,
  `[embedding]`, `[chunking]`, `[langextract]` byte-for-byte. Normalize whitespace to a clean
  contiguous file (SRS FR-1.4); do not reorder/reformat kept sections.

## Change 2 — `cfg/nemo_specs.toml`: mirror the identical structural edit

Per SRS FR-1.2 / FR-4: same header line, same DELETE list at the same line ranges (identical
143-line layout — verified). **Preserve** the value-level differences (SRS FR-2.4):
`[general].output_dir = ./data/specs_20260516`, `data_dir = ./rawdata-pubs/parsed-specs`,
`[llm].model = gpt-oss:120b#…`, `[chunking].relevance_concurrency = 2`. Do not homogenize the
two files; the only post-edit differences are those value rows + existing inline comments.

## Critical files

- `cfg/nemo.toml` (edit — Change 1)
- `cfg/nemo_specs.toml` (edit — Change 2; mirror)
- **No `.py` changes** (SRS FR-5.1). Note: `cfg/nemo_specs.toml` may be git-untracked in this
  checkout (`??` in status) — confirm tracking before relying on `git diff --stat` (SRS §8 R6).

## Prerequisites

1. `.venv/bin/python` interpreter.
2. Both files at the audited 143-line layout (re-locate DELETE blocks by table header if drifted
   — SRS §7.2).
3. Ollama reachable at `localhost:11434` for the §Verification step-2 chunk smoke (import-time
   `Embedder`); non-empty `OPENAI_API_KEY`/`GOOGLE_API_KEY` in env/`.env` for the import gate
   (inert on the `--chunk-only` path, no egress). SRS §7.3.
4. A 1-doc corpus (`small_corpus/` or one copied `.md`) for the smoke.

## Verification

Per SRS §9 (do not duplicate — summary):
1. `.venv/bin/python -c "from aisa.utils import files; print(sorted(files.read_toml('cfg/nemo.toml')))"`
   → expect exactly `['chunking', 'embedding', 'general', 'langextract', 'llm']`; repeat for
   `cfg/nemo_specs.toml` (SRS FR-2.1, FR-4.1 — the core gate).
2. 1-doc scratch smoke:
   `.venv/bin/python _nemo.py --chunk-only --cfg cfg/nemo.toml --input_dir <small_corpus>
   --output_dir /tmp/cfgtest` → creates `/tmp/cfgtest/doc-chunks_*_random_logical/`, no
   `KeyError`/Pydantic error (Ollama must be reachable). Optional parity repeat with
   `--cfg cfg/nemo_specs.toml` into a scratch `--output_dir`.
3. Optional: run the `pipeline-smoke-runner` agent for the full Route B chain
   (`--sdg-logical → extract_artifacts → generate-qa → self-check`) on a 1-doc subset —
   confidence check, not a gate (SRS §9 full-run note).
4. `git diff --stat` (+ `git status` for the untracked caveat) shows only the two TOMLs (and the
   two new `plans/*.md`) changed — no `.py` (SRS FR-5.1).

## Risks

Compressed — full table in SRS §8. Biggest: a future code path / sibling tool reading a removed
section (R1 — mitigated by the §2 zero-reader grep, the FR-3 header contract, and the §9
`read_toml` kept-set assertion). Also: clipping a kept `[llm]`/`[embedding]` key → Pydantic
`TypeError` (R2 — caught loudly by the §9 step-2 `--chunk-only` smoke); cross-file structural
drift (R4); pre-existing `extract_artifacts.py --cfg cfg/nemo_specs.toml` `KeyError` trap, not
introduced here (R3); `cfg/nemo_specs.toml` possibly git-untracked (R6).
