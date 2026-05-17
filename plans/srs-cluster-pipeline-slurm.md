# Software Requirements Specification: Combined Cluster Slurm Job for the `random_logical` Pipeline

**Feature:** A single Slurm batch script (`cluster/logical-chunks-qa-sc-spec~.slurm`) that runs the
full four-stage `random_logical` Ollama pipeline (chunk → artifacts → QA → self-check) on the
PennDOT specs corpus with `gpt-oss:120b`, in one job, modelled on
`self-check/cluster/self-check-qa.slurm`.
**Component:** `nvidia-sdq_custom`
**Version:** 0.1 (draft)
**Status:** Proposed
**Companion plan:** `plans/plan-cluster-pipeline-slurm.md`
**Parent SRS:** `plans/srs-ollama-random-logical-pipeline.md` (pipeline + code this job orchestrates)

---

## 1. Introduction

### 1.1 Purpose
Define the requirements for a self-contained cluster Slurm job that executes the entire
local-Ollama `random_logical` pipeline end-to-end for the full 24-doc specs corpus on
`gpt-oss:120b`, producing the LLM-evaluated question set with **zero OpenAI/Gemini egress**. The
parent SRS specifies the code/configs; this SRS specifies only the cluster orchestration job and
the cluster-side prerequisites that the local smoke did not exercise.

### 1.2 Scope

**In scope**
- One new script `cluster/logical-chunks-qa-sc-spec~.slurm` running stages 1–4 sequentially in a
  single Slurm allocation, against `PROJ=/vast/lkhazanovich/igs18/llm/nvidia-sdq_custom`.
- Cluster-essential additions over the single-stage templates: Ollama readiness wait, defensive
  idempotent `ollama pull`, mandatory `OPENAI_API_KEY`/`GOOGLE_API_KEY` export (stage-1 import
  gate), fail-fast (`set -eo pipefail`), per-stage logs, sequential stage handoff.
- **The single config edit:** `cfg/nemo_specs.toml:55` `[embedding].model`
  `nomic-embed-text:latest`→`nomic-embed-text:v1.5`. The four `_specs` configs **already**
  target `gpt-oss:120b` (`cfg/nemo_specs.toml:45,74`; `extract_artifacts_specs.toml:12,15`;
  `generate-qa_specs.toml:22-23`; `self-check/self-check-qa_specs.toml:16,27`) — **verify, do
  NOT re-edit**; the parent SRS still shows `gpt-oss:20b`/`relevance_concurrency` edits that are
  *already applied* and now stale for the cluster.
- Documentation of cluster prerequisites (env deps, model pulls, HF cache, corpus deploy).

**Out of scope**
- Code changes to the pipeline (done & specified in the parent SRS; this job consumes it as-is).
- Per-stage slurm jobs (superseded by this combined job; the existing `cluster/generate-qa.slurm`
  and `self-check/cluster/self-check-qa.slurm` remain for the techbriefs/standalone flows).
- Multi-node / multi-GPU; checkpoint logic beyond the pipeline's existing file-cache idempotency.
- Running `sbatch` or provisioning the cluster env (operator actions; see §6).

---

## 2. Background

The pipeline and its Ollama refactor are implemented and locally smoke-validated (parent SRS,
status: 4-stage local smoke PASSED, zero egress). Existing cluster jobs each run a **single**
stage from a per-stage working dir (`/vast/.../llm/qa-gen/`, `/vast/.../llm/self-check/`) with this
shape: SBATCH (h200, `gpu:1`, `--time=1-23:59:00`) → `module purge` → `cd <ollama
0.22.0 dir>` → dynamic free `$PORT` → `export OLLAMA_HOST=http://127.0.0.1:$PORT` + `OLLAMA_MODELS`
+ cuda/gcc + `OLLAMA_CONTEXT_LENGTH/NUM_PARALLEL/MAX_QUEUE/NO_CLOUD/FLASH_ATTENTION` →
`./bin/ollama serve &` → `source activate /vast/lkhazanovich/igs18/envs/qac-env` →
`cd <stage dir>` → `python <script> --config <toml> --host localhost --port $PORT` →
`crc-job-stats`.

Cluster facts established by audit:
- `cluster/models.txt` includes `gpt-oss:120b` and `nomic-embed-text:v1.5` (**not** `:latest`).
- `cluster/pull_models.sh` pulls `models.txt` via `./<ver>/bin/ollama pull` (idempotent).
- `generate-qa.slurm`'s `LD_LIBRARY_PATH` has a doubled `qac-env/qac-env` segment (likely a typo);
  `self-check-qa.slurm`'s single-segment path is the correct form to copy.
- Stages 3–4 have run on the cluster; **stages 1–2 (`_nemo.py`, `extract_artifacts.py`) have
  not** — `qac-env` may lack their dependencies.
- Parent-SRS facts that bite on the cluster (the parent has a **flat §7 list**, not
  sub-sections — cite items, not `§7.3`): import-key gate (parent **FR-1.3 / §7 item 2** —
  `_nemo.py` import requires non-empty `OPENAI_API_KEY`/`GOOGLE_API_KEY`); runtime `Embedder`
  tag-match else HF fallback (parent **§7 item 3**); import-time HF `all-MiniLM-L6-v2` (parent
  **§7 item 4**); FR-8 (`_nemo.py` relevance + `ollama_api` + `ChatOllama` honor `$OLLAMA_HOST`;
  `extract_artifacts.py` `--host/--port` **beats** `$OLLAMA_HOST`).
- Two cluster templates differ: `self-check-qa.slurm:29-30` = `OLLAMA_CONTEXT_LENGTH=65536`/
  `NUM_PARALLEL=8`; `generate-qa.slurm:29-30` = `32768`/`16`. This job follows the **self-check**
  template (65536/8); it **supersedes** parent SRS §7 item 6's stale generic `32768/16`.

---

## 3. Functional Requirements

### FR-1 — Single-job, sequential 4-stage execution
- **FR-1.1** The script SHALL run, in order and fail-fast (`set -eo pipefail`, placed after
  `ollama serve &`): (1) `_nemo.py --sdg-logical --cfg cfg/nemo_specs.toml`;
  (2) `extract_artifacts.py --cfg extract_artifacts_specs.toml --host localhost --port $PORT`;
  (3) `generate-qa.py --config generate-qa_specs.toml --host localhost --port $PORT`;
  (4) from `$PROJ/self-check`, `self-check-qa.py --config ./self-check-qa_specs.toml --host
  localhost --port $PORT`.
- **FR-1.2** Stages 1–3 run with CWD `$PROJ`; stage 4 with CWD `$PROJ/self-check` (its config
  paths are relative to that dir). After stage 4, CWD returns to `$PROJ` for `crc-job-stats`.
- **FR-1.3** Each stage's stdout+stderr SHALL append to a distinct
  `$PROJ/logs/0N-<stage>.log`; the Ollama server log to `$PROJ/logs/ollama-spec.log`.
- **FR-1.4** Stage 1 SHALL NOT pass `--host/--port` (it has none); it is the **only** stage that
  uses `$OLLAMA_HOST`. Stages 2–4 SHALL pass `--host localhost --port $PORT`, which **wins** their
  endpoint precedence (CLI > config > `$OLLAMA_HOST` > default) — so `$OLLAMA_HOST` is a fallback,
  not their source. `localhost` ≡ the server's `127.0.0.1` loopback bind; use `--host 127.0.0.1`
  only if a node maps `localhost`→IPv6.

### FR-2 — Ollama bring-up (template-derived)
- **FR-2.1** SHALL reproduce the template's bring-up: `module purge`; `cd
  /vast/lkhazanovich/igs18/ollama/0.22.0/`; `module load python/ondemand-jupyter-python3.11`
  (needed for the `$PORT` socket snippet); dynamic free `$PORT` (+ guard `[ -n "$PORT" ]`); `export
  OLLAMA_HOST=http://127.0.0.1:$PORT`, `OLLAMA_MODELS=/vast/lkhazanovich/igs18/ollama/models`;
  `module load cuda/12.9.0 gcc/12.2.0`; `OLLAMA_CONTEXT_LENGTH=65536`,
  `OLLAMA_NUM_PARALLEL=8`, `OLLAMA_MAX_QUEUE=512`, `OLLAMA_NO_CLOUD=1`,
  `OLLAMA_FLASH_ATTENTION=1`; `./bin/ollama serve &>> $PROJ/logs/ollama-spec.log &`.
- **FR-2.2** SHALL activate `source activate /vast/lkhazanovich/igs18/envs/qac-env` and set
  `LD_LIBRARY_PATH` using the **single-segment** `.../qac-env/x86_64-conda-linux-gnu/lib/` form
  (not `generate-qa.slurm`'s doubled path).
- **FR-2.3** SHALL wait for readiness before stage 1: poll `curl -sf $OLLAMA_HOST/api/tags` (≤120×
  5 s); abort with non-zero exit if never ready.
- **FR-2.4** SHALL defensively `./bin/ollama pull gpt-oss:120b` and
  `./bin/ollama pull nomic-embed-text:v1.5` (idempotent; cheap if present) before the stages.

### FR-3 — Stage-1 import gate (parent FR-1.3 / §7 item 2)
- **FR-3.1** Before stage 1 the script SHALL `export OPENAI_API_KEY` and `GOOGLE_API_KEY` to
  **non-empty** values if unset (e.g. `${OPENAI_API_KEY:-sk-cluster-ollama-noegress}`). These are
  never used for network calls (all models are Ollama) but are required for `_nemo.py` to import
  (`aisa/gen/providers.py`). Bogus values are sufficient and preserve zero-egress.

### FR-4 — Config alignment (embedding tag)
- **FR-4.1** `cfg/nemo_specs.toml:55` `[embedding].model` SHALL be `nomic-embed-text:v1.5` (a tag
  in `cluster/models.txt`), not `nomic-embed-text:latest` (still on disk). The runtime `Embedder`
  (constructed even for `--sdg-logical`, `_nemo.py:843`; also at import via the
  `QAGenerator.__init__` default arg) matches the model string against `ollama list`; a miss
  falls to `HuggingFaceEmbeddings("nomic-embed-text:latest")` (invalid HF id) and aborts stage 1.
- **FR-4.2** This is the **only** config edit. The four `_specs` configs already target
  `gpt-oss:120b` with cluster-tuned concurrency — **verify, do not re-edit** (the parent SRS's
  `gpt-oss:20b`/`relevance_concurrency` edits are already applied and now stale here).

### FR-5 — Idempotent resume
- **FR-5.1** The job SHALL rely on the pipeline's existing per-doc/per-file caching for resume: a
  job hitting the wall-time limit is recovered by resubmitting the same `sbatch` (completed docs
  and outputs are cache-hit and skipped). No additional checkpoint logic is required.

---

## 4. External Interfaces

- **Slurm:** `sbatch cluster/logical-chunks-qa-sc-spec~.slurm`; SBATCH directives per template
  (`--cluster=gpu --partition=h200 --gres=gpu:1 --time=1-23:59:00 --nodes=1
  --ntasks-per-node=1`, mail, `--job-name="lc-qa-sc-spec"`).
- **Ollama:** local server at `$OLLAMA_HOST=http://127.0.0.1:$PORT`; `$OLLAMA_MODELS` store;
  CLI `./bin/ollama serve|pull`.
- **Pipeline scripts:** as in the parent SRS; consumed via the four `_specs` configs unchanged
  (already cluster-ready — `gpt-oss:120b`). `extract_artifacts_specs.toml` omits `ollama_host`,
  but stage 2 passes `--host/--port` which **wins** precedence — `$OLLAMA_HOST` is only the
  fallback if CLI is absent. **Only stage 1** actually resolves the endpoint via `$OLLAMA_HOST`.
- **Conda env:** `/vast/lkhazanovich/igs18/envs/qac-env`.

---

## 5. The Script (normative content)

`cluster/logical-chunks-qa-sc-spec~.slurm`:

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1-23:59:00
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=igs18@pitt.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --cluster=gpu
#SBATCH --partition=h200
#SBATCH --job-name="lc-qa-sc-spec"

module purge
cd /vast/lkhazanovich/igs18/ollama/0.22.0/

module load python/ondemand-jupyter-python3.11
PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
[ -n "$PORT" ] || { echo "failed to obtain a free port" >&2; exit 1; }
ADDRESS=$(hostname -I | awk '{print $1}')

PROJ=/vast/lkhazanovich/igs18/llm/nvidia-sdq_custom
LOGS=$PROJ/logs
mkdir -p "$LOGS"

export OLLAMA_MODELS="/vast/lkhazanovich/igs18/ollama/models"
export OLLAMA_HOST="http://127.0.0.1:$PORT"

module load cuda/12.9.0
module load gcc/12.2.0

export OLLAMA_CONTEXT_LENGTH=65536
export OLLAMA_NUM_PARALLEL=8
export OLLAMA_MAX_QUEUE=512
export OLLAMA_NO_CLOUD=1
export OLLAMA_FLASH_ATTENTION=1
./bin/ollama serve &>> "$LOGS/ollama-spec.log" &

source activate /vast/lkhazanovich/igs18/envs/qac-env
export LD_LIBRARY_PATH=/vast/lkhazanovich/igs18/envs/qac-env/x86_64-conda-linux-gnu/lib/:$LD_LIBRARY_PATH

echo "-- host is $ADDRESS"
echo "-- port is $PORT"

for i in $(seq 1 120); do
  if curl -sf "$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then READY=1; break; fi
  sleep 5
done
[ "${READY:-}" = 1 ] || { echo "ollama not ready on $OLLAMA_HOST" >&2; exit 1; }

./bin/ollama pull gpt-oss:120b          &>> "$LOGS/ollama-spec.log"
./bin/ollama pull nomic-embed-text:v1.5 &>> "$LOGS/ollama-spec.log"

export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-cluster-ollama-noegress}"
export GOOGLE_API_KEY="${GOOGLE_API_KEY:-cluster-ollama-noegress}"

# Pre-warm: force the ~65 GB gpt-oss:120b load NOW. The first timed stage is
# stage-1 relevance, which uses the OpenAI SDK's 600 s default (no timeout= in
# _nemo.py); a cold load exceeding it raises APITimeoutError that the relevance
# except→score=1.0 silently swallows (parent §7 R8 / §8 R8 below).
curl -sf "$OLLAMA_HOST/api/generate" \
  -d '{"model":"gpt-oss:120b","prompt":"ready","stream":false}' \
  &>> "$LOGS/ollama-spec.log" || echo "pre-warm returned non-zero (continuing)" >&2

set -eo pipefail
trap 'crc-job-stats' EXIT          # emit job stats on EVERY exit path (incl. a failed stage)
cd "$PROJ"

echo "[stage 1] _nemo.py --sdg-logical (uses \$OLLAMA_HOST)"
python _nemo.py --sdg-logical --cfg cfg/nemo_specs.toml &>> "$LOGS/01-chunk.log" \
  || { echo "[stage 1] FAILED (exit $?)" >&2; exit 11; }

echo "[stage 2] extract_artifacts.py  (flag is --cfg, NOT --config)"
python extract_artifacts.py --cfg extract_artifacts_specs.toml \
  --host localhost --port "$PORT" &>> "$LOGS/02-artifacts.log" \
  || { echo "[stage 2] FAILED (exit $?)" >&2; exit 12; }

echo "[stage 3] generate-qa.py  (flag is --config)"
python generate-qa.py --config generate-qa_specs.toml \
  --host localhost --port "$PORT" &>> "$LOGS/03-qa.log" \
  || { echo "[stage 3] FAILED (exit $?)" >&2; exit 13; }

echo "[stage 4] self-check  (--config; run from self-check/)"
cd "$PROJ/self-check"
python self-check-qa.py --config ./self-check-qa_specs.toml \
  --host localhost --port "$PORT" &>> "$LOGS/04-selfcheck.log" \
  || { echo "[stage 4] FAILED (exit $?)" >&2; exit 14; }

cd "$PROJ"
# crc-job-stats runs via the EXIT trap above (covers success and any failed stage)
```

`set -eo pipefail` omits `-u` (conda/module scripts reference unset vars) and is placed after
`ollama serve &`; the pre-`set -e` `$PORT` guard and the readiness loop's explicit `exit 1`
catch the common early failures. Each stage uses `… || { echo FAILED; exit 1N; }` for clear
per-stage attribution, and `trap 'crc-job-stats' EXIT` guarantees the resource report on
success **and** on a failed stage (a bare trailing `crc-job-stats` would be skipped by `set -e`).
The pre-warm `curl /api/generate` forces the ~65 GB `gpt-oss:120b` load before any timed stage
(see §8 R8). `OLLAMA_NUM_PARALLEL=8` follows the 120b-class self-check template and caps the
`_specs` concurrency (`chunk_concurrency=8`, `max_concurrent_qa=20`,
`max_concurrent_questions=16`; `relevance_concurrency=2` unaffected) — throughput-only, but see
§8 R9 for the VRAM trade-off of 8 concurrent 64 K-context 120b slots on one H200.

---

## 6. Prerequisites (operator, before `sbatch`)

1. Full repo tree at `PROJ` incl. the refactored code, the four `_specs` configs, `prompts/`,
   and `rawdata-pubs/parsed-specs/*.md` (24).
2. `pip install -r reqs.txt` into `qac-env` (stages 1–2 deps: `langextract`, `langchain_ollama`,
   `langchain_openai`, `langchain_google_genai`, `tiktoken`, `faiss-cpu`, …).
3. FR-4 config edit applied.
4. HF `sentence-transformers/all-MiniLM-L6-v2` cached in a shared `HF_HOME` if compute nodes are
   offline (add `export HF_HOME=<path>` to the script if not the default).
5. `gpt-oss:120b` + `nomic-embed-text:v1.5` pre-pulled (`cluster/pull_models.sh`); the in-job
   pulls are a safety net, not a substitute (cold 120b ≈ 65 GB).

---

## 7. Acceptance Criteria / Verification

1. **Login-node pre-flight:** `conda activate qac-env`;
   `python -c "import langextract, ollama, langchain_ollama, tiktoken"`;
   `ls $PROJ/rawdata-pubs/parsed-specs/*.md | wc -l` == 24;
   `grep nomic-embed $PROJ/cfg/nemo_specs.toml` shows `:v1.5`;
   `./bin/ollama list | grep -E "gpt-oss:120b|nomic-embed-text:v1.5"`.
2. `sbatch cluster/logical-chunks-qa-sc-spec~.slurm`; monitor `$PROJ/logs/0{1..4}-*.log`.
3. Per-stage (mirrors parent SRS §9): S1 `*-logic-ctx.json` for the corpus + no
   `defaulting to score=1.0` in `01-chunk.log`; S2 `*-logic-artifacts.json` with `errors` null,
   non-empty `extractions`, `chunk_signals`; S3 non-empty Q/A/context+citations; S4
   `self-check-output-specs/self-check-qa-results.json` with `evaluation ∈ {0,0.5,1}`.
4. **Zero-egress:** with bogus keys, grep all stage logs for `AuthenticationError` /
   `api.openai.com` / `generativelanguage` → none (all traffic to `$OLLAMA_HOST`).
5. **Acceptance:** all four stages exit 0 in one allocation (or across resubmits via FR-5), the
   deliverable JSON exists for the corpus, zero egress.

---

## 8. Risks & Mitigations

| # | Risk | Mitigation |
|---|---|---|
| R1 | Wall-clock > `1-23:59:00` (uncapped 24-doc, 120b, 4 stages serial) | FR-5: stages file-cached/idempotent — resubmit same `sbatch` resumes; monitor `crc-job-stats` |
| R2 | `qac-env` lacks stage 1–2 deps → `ImportError` | Prereq 2 + pre-flight import check (§7.1) |
| R3 | Embedding-tag miss / import-time HF model → stage-1 crash | FR-4 + prereq 4; defensive pull (FR-2.4) |
| R4 | Compute node offline → in-job `ollama pull` / HF download fails | Prereqs 4–5 (pre-pull, pre-cache on login node) |
| R5 | Missing `OPENAI_API_KEY`/`GOOGLE_API_KEY` → stage-1 import abort | FR-3 (mandatory bogus-key export) |
| R6 | `NUM_PARALLEL=8` caps `chunk_concurrency=8`/`max_concurrent_qa=20`/`max_concurrent_questions=16` (`relevance_concurrency=2` unaffected) | Throughput-only, not incorrect — but do **not** blindly raise; see R9 |
| R7 | `generate-qa.slurm` LD_LIBRARY_PATH typo copied | FR-2.2 mandates the single-segment self-check form |
| R8 | **Cold-load → silent keep-all (parent §7 R8).** First `gpt-oss:120b` request loads ~65 GB; stage-1 relevance uses the OpenAI-SDK **600 s** default (no `timeout=` at `_nemo.py:189/395`) and its `except→score=1.0` swallows `APITimeoutError` ⇒ relevance filter silently OFF. Stage-2 span (`ollama_timeout=600`) is also borderline if still cold | §5 **pre-warm** `curl /api/generate` before the stages; §7 assertion "no `defaulting to score=1.0` in `01-chunk.log`" is a **hard gate**. *Optional code addendum (parent SRS, deferred):* `AsyncOpenAI(..., timeout=1800)` at `_nemo.py:189` |
| R9 | **VRAM.** `NUM_PARALLEL=8 × OLLAMA_CONTEXT_LENGTH=65536 × gpt-oss:120b` on one H200 may KV-cache-OOM/thrash (the self-check template's 8 was sized for short judge prompts, not 64 K stage-2/3 contexts) | Treat 8 as a ceiling only if it fits; consider `OLLAMA_NUM_PARALLEL=2–4`; watch `$LOGS/ollama-spec.log` for OOM on the first scale-up doc |

---

## 9. Future Work (out of scope)

- Parameterize `PROJ`/partition/time via script args or `#SBATCH` overrides.
- A fast cluster smoke before the full 24-doc run: only `self-check-qa.py` has `--limit`
  (`:200`); stages 1–3 have none — subset via a throwaway 1-doc `--input_dir` for stage 1
  (parent SRS §9 mechanism) then let stages 2–4 glob just that doc.
- Split into a Slurm job array (per-doc parallelism) if single-job wall-clock proves insufficient.
