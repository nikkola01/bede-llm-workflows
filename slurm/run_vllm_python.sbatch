#!/bin/bash
#SBATCH --job-name=vllm-infer
#SBATCH --partition=ghtest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH -A <project_ID>
#SBATCH --output=vllm_infer_%j.out
#SBATCH --error=vllm_infer_%j.err

set -euo pipefail

echo "===== JOB START ====="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Using GPU:"
nvidia-smi

# -----------------------------
# User-editable section
# -----------------------------
PROJECT=""
USER_NAME=""

BASE="/nobackup/projects/${PROJECT}/${USER_NAME}/modules/vllm26"
CONTAINER_DIR="/nobackup/projects/${PROJECT}/${USER_NAME}/containers"

IMAGE="${CONTAINER_DIR}/vllm-ag2-26.01.1.sif"

HOME_DIR="${BASE}/home"
WORK_DIR="${BASE}/workspace"
CACHE_DIR="${WORK_DIR}/.cache"
PY_FILE="${WORK_DIR}/tt2.py"

MODEL="Qwen/Qwen2.5-32B-Instruct"
PROMPT="Explain the difference between precision and accuracy in scientific experiments."
TEMPERATURE="0.7"
MAXTOKENS="3000"
# -----------------------------

if [[ ! -f "$IMAGE" ]]; then
    echo "ERROR: container image $IMAGE not found"
    exit 1
fi

mkdir -p "$HOME_DIR" "$WORK_DIR" "$CACHE_DIR"
chmod -R a+rwx "$BASE"

echo "===== WRITING PYTHON FILE ====="

cat > "$PY_FILE" <<'PYEOF'
from vllm import LLM, SamplingParams
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--prompt", required=True)
parser.add_argument("--maxtokens", type=int, default=3000)
parser.add_argument("--temperature", type=float, default=0.7)
args = parser.parse_args()

sampling_params = SamplingParams(
    temperature=args.temperature,
    max_tokens=args.maxtokens
)

llm = LLM(model=args.model, tensor_parallel_size=1)
outputs = llm.generate([args.prompt], sampling_params)

for out in outputs:
    print("\n===== RESPONSE =====\n")
    print(out.outputs[0].text)
PYEOF

# Verify the file was really written
if [[ ! -s "$PY_FILE" ]]; then
    echo "ERROR: Python file was not created correctly: $PY_FILE"
    exit 1
fi

echo "Python file created:"
ls -l "$PY_FILE"
echo "----- preview -----"
head -n 10 "$PY_FILE"
echo "-------------------"

echo "===== RUNNING INFERENCE ====="
echo "Model: $MODEL"
apptainer exec --nv \
  --bind "$WORK_DIR:/workspace" \
  --home "$HOME_DIR:/users/$USER_NAME" \
  --env HF_HOME=/workspace/.cache/huggingface \
  --env XDG_CACHE_HOME=/workspace/.cache \
  --env FLASHINFER_WORKSPACE_DIR=/users/$USER_NAME/.cache/flashinfer \
  --env TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torchinductor \
  --env TRITON_CACHE_DIR=/workspace/.cache/triton \
  --env VLLM_DISABLE_COMPILE=1 \
  "$IMAGE" \
  python /workspace/tt1.py \
    --model "$MODEL" \
    --prompt "$PROMPT" \
    --maxtokens "$MAXTOKENS" \
    --temperature "$TEMPERATURE"echo "Prompt: $PROMPT"

echo "===== JOB END ====="
date
