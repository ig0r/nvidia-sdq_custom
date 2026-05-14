#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1-23:59:00
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=igs18@pitt.edu    
#SBATCH --mail-type=BEGIN,END,FAIL 
#SBATCH --gres=gpu:1
#SBATCH --cluster=gpu
#SBATCH --partition=h200
#SBATCH --job-name="242-sc"

module purge

#cd /ix1/lkhazanovich/igs18/ollama/0.16.1/
cd /ix1/lkhazanovich/igs18/ollama/0.17.6/

module load python/ondemand-jupyter-python3.11
PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
ADDRESS=$(hostname -I | awk '{print $1}')

export OLLAMA_MODELS="/ix1/lkhazanovich/igs18/ollama/models"
export OLLAMA_HOST="http://127.0.0.1:$PORT"

touch /ix1/lkhazanovich/igs18/llm/procs/selfcheck-eval/logs/ollama-pub242.log

module load cuda/12.9.0
module load gcc/12.2.0

export OLLAMA_CONTEXT_LENGTH=65536
export OLLAMA_NUM_PARALLEL=8
export OLLAMA_MAX_QUEUE=512
export OLLAMA_NO_CLOUD=1
export OLLAMA_FLASH_ATTENTION=1
./bin/ollama serve &>> /ix1/lkhazanovich/igs18/llm/procs/selfcheck-eval/logs/ollama-pub242.log &

# activating environment
source activate /ix1/lkhazanovich/igs18/envs/proc-env
export LD_LIBRARY_PATH=/ix1/lkhazanovich/igs18/envs/proc-env/x86_64-conda-linux-gnu/lib/:$LD_LIBRARY_PATH

echo "-- host is $ADDRESS"
echo "-- port is $PORT"

cd /ix1/lkhazanovich/igs18/llm/procs/selfcheck-eval/
touch ./logs/inference-pub242.log

python get-categorized-questions-eval-async2.py --config get-categorized-questions-eval-async2-pub242.toml --host "localhost" --port $PORT &>> ./logs/inference-pub242.log

crc-job-stats
