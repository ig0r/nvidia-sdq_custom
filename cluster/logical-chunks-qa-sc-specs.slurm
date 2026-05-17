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

PROJ=/vast/lkhazanovich/igs18/llm/general-qa-sc
LOGS=$PROJ/logs
mkdir -p "$LOGS"

export OLLAMA_MODELS="/vast/lkhazanovich/igs18/ollama/models"
export OLLAMA_HOST="http://127.0.0.1:$PORT"

module load cuda/12.9.0
module load gcc/12.2.0

export OLLAMA_CONTEXT_LENGTH=32768
export OLLAMA_NUM_PARALLEL=8
export OLLAMA_MAX_QUEUE=512
export OLLAMA_NO_CLOUD=1
export OLLAMA_FLASH_ATTENTION=1
./bin/ollama serve &>> "$LOGS/ollama-spec.log" &

source activate /vast/lkhazanovich/igs18/envs/general-qa-sc-env
export LD_LIBRARY_PATH=/vast/lkhazanovich/igs18/envs/general-qa-sc-env/x86_64-conda-linux-gnu/lib/:$LD_LIBRARY_PATH

echo "-- host is $ADDRESS"
echo "-- port is $PORT"

for i in $(seq 1 120); do
  if curl -sf "$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then READY=1; break; fi
  sleep 5
done
[ "${READY:-}" = 1 ] || { echo "ollama not ready on $OLLAMA_HOST" >&2; exit 1; }

#./bin/ollama pull gpt-oss:120b          &>> "$LOGS/ollama-spec.log"
#./bin/ollama pull nomic-embed-text:v1.5 &>> "$LOGS/ollama-spec.log"

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
python _nemo.py --sdg-logical --cfg ./nemo_specs.toml &>> "$LOGS/01-chunk.log" \
  || { echo "[stage 1] FAILED (exit $?)" >&2; exit 11; }

echo "[stage 2] extract_artifacts.py  (flag is --cfg, NOT --config)"
python extract_artifacts.py --cfg ./extract_artifacts_specs.toml \
  --host localhost --port "$PORT" &>> "$LOGS/02-artifacts.log" \
  || { echo "[stage 2] FAILED (exit $?)" >&2; exit 12; }

echo "[stage 3] generate-qa.py  (flag is --config)"
python generate-qa.py --config ./generate-qa_specs.toml \
  --host localhost --port "$PORT" &>> "$LOGS/03-qa.log" \
  || { echo "[stage 3] FAILED (exit $?)" >&2; exit 13; }

echo "[stage 4] self-check  (--config; run from self-check/)"
python self-check-qa.py --config ./self-check-qa_specs.toml \
  --host localhost --port "$PORT" &>> "$LOGS/04-selfcheck.log" \
  || { echo "[stage 4] FAILED (exit $?)" >&2; exit 14; }

cd "$PROJ"
# crc-job-stats runs via the EXIT trap above (covers success and any failed stage)
