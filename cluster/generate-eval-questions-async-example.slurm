#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1-23:59:00
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=igs18@pitt.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --cluster=gpu
#SBATCH --partition=h200
#SBATCH --job-name="gen-eval-q"

module purge

cd /ix1/lkhazanovich/igs18/ollama/0.16.1/

module load python/ondemand-jupyter-python3.11
PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
ADDRESS=$(hostname -I | awk '{print $1}')

export OLLAMA_MODELS="/ix1/lkhazanovich/igs18/ollama/models"
export OLLAMA_HOST="http://127.0.0.1:$PORT"

touch /ix1/lkhazanovich/igs18/llm/generate-eval-questions/logs/ollama-gen-eval-q.log

module load cuda/12.9.0
module load gcc/12.2.0

export OLLAMA_CONTEXT_LENGTH=32768
export OLLAMA_NUM_PARALLEL=16
export OLLAMA_MAX_QUEUE=512
export OLLAMA_NO_CLOUD=1
export OLLAMA_FLASH_ATTENTION=1
./bin/ollama serve &>> /ix1/lkhazanovich/igs18/llm/generate-eval-questions/logs/ollama-gen-eval-q.log &

# activating environment
source activate /ix1/lkhazanovich/igs18/envs/proc-env
export LD_LIBRARY_PATH=/ix1/lkhazanovich/igs18/envs/proc-env/x86_64-conda-linux-gnu/lib/:$LD_LIBRARY_PATH

echo "-- host is $ADDRESS"
echo "-- port is $PORT"

cd /ix1/lkhazanovich/igs18/llm/generate-eval-questions/
touch ./logs/inference-gen-eval-q.log

python generate-eval-questions-async.py --config generate-eval-questions-async.toml --host "localhost" --port $PORT &>> ./logs/inference-gen-eval-q.log

crc-job-stats
