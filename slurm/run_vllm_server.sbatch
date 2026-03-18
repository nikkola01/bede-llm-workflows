#!/bin/bash
#SBATCH --job-name=vllm-server
#SBATCH --partition=ghtest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH -A <project_ID>
#SBATCH --output=vllm_server_%j.out
#SBATCH --error=vllm_server_%j.err

set -euo pipefail

echo "===== JOB START ====="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "GPU:"
nvidia-smi

IMAGE="vllm-ag2-26.01.1.sif"
if [[ ! -f "$IMAGE" ]]; then
  echo "ERROR: container image $IMAGE not found"
  exit 1
fi

# --------- writable cache/work dirs on /nobackup ---------
BASE="/nobackup/projects/<project_ID>/<user_ID>/modules/vllm26_server"
HOME_DIR="$BASE/home"
WORK_DIR="$BASE/workspace"

mkdir -p "$HOME_DIR" "$WORK_DIR"
chmod -R a+rwx "$BASE"

# Pick a port (change if you like)
PORT=8080

# Model to serve (small public model for testing)
MODEL="facebook/opt-125m"

echo "===== STARTING vLLM SERVER ====="
echo "Model: $MODEL"
echo "Port:  $PORT"
echo "Logs:  vllm_server_${SLURM_JOB_ID}.out / .err"
echo
echo "After it starts, test from the SAME NODE with:"
echo "  curl http://127.0.0.1:${PORT}/v1/models"
echo

apptainer exec --nv \
  --bind "$WORK_DIR:/workspace" \
  --home "$HOME_DIR:/users/nin6" \
  --env HF_HOME=/workspace/.cache/huggingface \
  --env XDG_CACHE_HOME=/workspace/.cache \
  --env FLASHINFER_WORKSPACE_DIR=/users/nin6/.cache/flashinfer \
  --env TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torchinductor \
  --env TRITON_CACHE_DIR=/workspace/.cache/triton \
  --env TERM=dumb \
  "$IMAGE" bash -lc "
set -eux
mkdir -p /workspace/.cache/huggingface
mkdir -p /workspace/.cache/torchinductor
mkdir -p /workspace/.cache/triton
mkdir -p /users/nin6/.cache/flashinfer

# Start OpenAI-compatible server
# --host 0.0.0.0 binds to all interfaces on the node (needed for tunneling)
vllm serve \"$MODEL\" \
  --host 0.0.0.0 \
  --port ${PORT} \
  --dtype float16 \
  --tensor-parallel-size 1
"

echo "===== JOB END ====="
date
