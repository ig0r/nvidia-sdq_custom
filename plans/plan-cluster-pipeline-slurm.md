# Plan: Combined cluster slurm job — full `random_logical` pipeline (specs, `gpt-oss:120b`)

**Companion SRS:** `plans/srs-cluster-pipeline-slurm.md`
**Parent:** `plans/srs-ollama-random-logical-pipeline.md` (the pipeline this job orchestrates)
**Status:** Proposed
**Deliverable:** `cluster/logical-chunks-qa-sc-spec~.slurm` (filename literal, incl. trailing `~`)

## Why

The Ollama refactor is implemented and locally smoke-validated (stages 1–4, zero egress). The full
24-doc specs corpus on `gpt-oss:120b` is a cluster job. Existing slurm jobs run **one** stage each
from per-stage dirs; this is a **single job running all 4 stages** sequentially, modelled on
`self-check/cluster/self-check-qa.slurm`.

## Decisions (user)

- Cluster project root `PROJ=/vast/lkhazanovich/igs18/llm/nvidia-sdq_custom` (one tree, all stages).
- Self-contained & defensive: ollama readiness wait, defensive `ollama pull`, fail-fast, per-stage logs.

## Job design (mirrors the template + adds the combined-run essentials)

SBATCH header copied from `self-check-qa.slurm` (h200, `gpu:1`, `--time=1-23:59:00`,
`job-name="lc-qa-sc-spec"`). Body: `module purge` → `cd /vast/lkhazanovich/igs18/ollama/0.22.0/`
→ dynamic free `$PORT` → `export OLLAMA_HOST=http://127.0.0.1:$PORT`, `OLLAMA_MODELS`, cuda/gcc
modules, `OLLAMA_CONTEXT_LENGTH=65536`/`NUM_PARALLEL=8`/… → `./bin/ollama serve &` →
`source activate /vast/lkhazanovich/igs18/envs/qac-env` (+ self-check.slurm `LD_LIBRARY_PATH`,
**not** the doubled path in `generate-qa.slurm`) → **readiness wait** (`curl $OLLAMA_HOST/api/tags`)
→ defensive `ollama pull gpt-oss:120b` + `nomic-embed-text:v1.5` → **export bogus-but-non-empty
`OPENAI_API_KEY`/`GOOGLE_API_KEY`** (parent FR-1.3 / §7 item 2 — `_nemo.py` import gate) →
**pre-warm** `curl /api/generate` (cold 65 GB load — SRS R8) → `set -eo pipefail` +
`trap 'crc-job-stats' EXIT` → `cd $PROJ` → 4 sequential stages, each
`&>> $PROJ/logs/0N-*.log || { echo FAILED; exit 1N; }`.

Stage commands:
1. `python _nemo.py --sdg-logical --cfg cfg/nemo_specs.toml` — no `--host/--port`; uses
   `$OLLAMA_HOST` (FR-8: relevance + `ollama_api` + `ChatOllama`).
2. `python extract_artifacts.py --cfg extract_artifacts_specs.toml --host localhost --port $PORT`
3. `python generate-qa.py --config generate-qa_specs.toml --host localhost --port $PORT`
4. `cd $PROJ/self-check && python self-check-qa.py --config ./self-check-qa_specs.toml --host localhost --port $PORT`

Full script content in SRS §5.

## Prerequisites (cluster, before `sbatch` — not done by this plan)

1. Full repo tree + 4 `_specs` configs + `prompts/` + `rawdata-pubs/parsed-specs/*.md` (24) at `PROJ`.
2. `pip install -r reqs.txt` into `qac-env` (stages 1–2 deps: `langextract`, `langchain_ollama`,
   aisa stack — never run on cluster before).
3. **The only config edit:** `cfg/nemo_specs.toml:55` `[embedding].model`
   `nomic-embed-text:latest` → `nomic-embed-text:v1.5` (cluster pulls `:v1.5`; tag-miss → HF
   fallback crashes stage 1). The four `_specs` configs are **already** `gpt-oss:120b` — verify,
   do NOT re-edit (parent SRS's `:20b` edits are applied & now stale here).
4. Pre-cache HF `sentence-transformers/all-MiniLM-L6-v2` (import-time, parent §7 item 4) into a
   shared `HF_HOME` if compute nodes are offline.
5. Pre-pull `gpt-oss:120b` (~65 GB) via `cluster/pull_models.sh` (job also pulls defensively).

## Critical files

- **Create:** `cluster/logical-chunks-qa-sc-spec~.slurm`.
- **Reference:** `self-check/cluster/self-check-qa.slurm`, `cluster/generate-qa.slurm`,
  `cluster/pull_models.sh`, `cluster/models.txt`.
- **Edit (prereq 3):** `cfg/nemo_specs.toml` `[embedding].model`.
- Consumed unchanged: `cfg/nemo_specs.toml`, `extract_artifacts_specs.toml`,
  `generate-qa_specs.toml`, `self-check/self-check-qa_specs.toml` (already cluster-ready —
  `gpt-oss:120b`, `ollama_host` omitted → resolves from `$OLLAMA_HOST`).

## Verification

Login-node pre-flight (imports, 24 `.md`, `:v1.5` in cfg, models present) → `sbatch` →
`tail -f $PROJ/logs/0{1..4}-*.log` → per-stage acceptance mirroring SRS-parent §9 (S1 `*-logic-ctx`
+ no `defaulting to score=1.0`; S2 non-empty `extractions`+`chunk_signals`; S3 non-empty Q/A;
S4 `evaluation ∈ {0,0.5,1}`) → zero-egress (no `AuthenticationError`/`api.openai.com` in logs).

## Risks

R1 wall-clock > `1-23:59:00` (24-doc 120b) — mitigated: stages idempotent/file-cached, resubmit
resumes. R2 qac-env missing stage 1–2 deps → ImportError (prereq 2). R3 embedding-tag miss / HF
model → stage-1 crash (prereq 3/4). R4 offline compute node breaks in-job pull/HF (prereq 4/5).
R8 cold 65 GB 120b load → stage-1 relevance OpenAI-SDK 600 s timeout silently swallowed (pre-warm
`curl` in §5; hard-gate `01-chunk.log`). R9 VRAM: `NUM_PARALLEL=8 × 65536 × 120b` on one H200 may
OOM — consider 2–4. Full table (R1–R9) in SRS §8.
