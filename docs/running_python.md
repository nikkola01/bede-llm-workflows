# Run a prompt through a Python file inside the vLLM container

This document explains how to run a **single prompt** by:

1. creating a small Python script inside the job
2. passing the model + prompt from the Slurm script
3. running that Python file inside the Apptainer container

This is a good pattern when you want something more flexible than a one-line command, but simpler than running a full server.

---

## 1  What this job is doing

This job:

- starts a Slurm allocation on `ghtest`
- checks the GPU
- defines a set of user-editable values
- writes a Python file into your workspace
- runs that Python file inside the container with `python ... --model ... --prompt ...`

So the `.sbatch` file acts as the launcher, and the Python file does the actual vLLM inference.

```bash
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
```

---

## 2  User-editable section

This is the part you are expected to change between runs.

### Paths and project settings

```bash
PROJECT=""
USER_NAME=""
BASE="/nobackup/projects/${PROJECT}/${USER_NAME}/modules/vllm26"
CONTAINER_DIR="/nobackup/projects/${PROJECT}/${USER_NAME}/containers"
IMAGE="${CONTAINER_DIR}/vllm-ag2-26.01.1.sif"
HOME_DIR="${BASE}/home"
WORK_DIR="${BASE}/workspace"
CACHE_DIR="${WORK_DIR}/.cache"
PY_FILE="${WORK_DIR}/tt2.py"
```

What these do:

- `PROJECT` and `USER_NAME` build the paths used in the rest of the script.
- `BASE` is your main working area on `/nobackup`.
- `CONTAINER_DIR` is where the `.sif` image lives.
- `IMAGE` is the full path to the container you want to run.
- `HOME_DIR` is the writable home directory inside the container.
- `WORK_DIR` is the writable workspace bound to `/workspace`.
- `CACHE_DIR` is where Hugging Face / Triton / Torch caches will go.
- `PY_FILE` is the Python script that the job writes before running.

### Model and prompt settings

```bash
MODEL="Qwen/Qwen2.5-32B-Instruct"
PROMPT="Explain the difference between precision and accuracy in scientific experiments."
TEMPERATURE="0.7"
MAXTOKENS="3000"
```

These are the most important user-editable values:

- `MODEL`
    
    The Hugging Face model name to load.
    
- `PROMPT`
    
    The text you want to send to the model.
    
- `TEMPERATURE`
    
    Controls randomness.
    
    - lower = more deterministic
    - higher = more varied
- `MAXTOKENS`
    
    Maximum number of tokens the model is allowed to generate.
    

---

## 3 The Python file being created

Your job writes a Python file with this logic:

```python
from vllm import LLM,SamplingParams
importargparse

parser=argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--prompt", required=True)
parser.add_argument("--maxtokens", type=int, default=3000)
parser.add_argument("--temperature", type=float, default=0.7)
args=parser.parse_args()

sampling_params=SamplingParams(
temperature=args.temperature,
max_tokens=args.maxtokens
)

llm=LLM(model=args.model)
outputs=llm.generate([args.prompt],sampling_params)

for out in outputs:
print("\n===== RESPONSE =====\n")
print(out.outputs[0].text)
```

### What this Python file does

- reads the command-line arguments
- loads the selected model with vLLM
- creates sampling parameters
- sends your prompt to the model
- prints the generated response into the Slurm log

This is a nice pattern because you do not need to edit Python every time you change the model or prompt. You only change the variables in the Slurm file.

---

## 4 Why the container options matter

This part is the same logic as in the earlier guides, but applied to Python inference.

```bash
echo "===== RUNNING INFERENCE ====="
echo "Model:$MODEL"
echo "Prompt:$PROMPT"

apptainer exec--nv \
--bind "$WORK_DIR:/workspace" \
--home "$HOME_DIR:/users/$USER_NAME" \
--env HF_HOME=/workspace/.cache/huggingface \
--env XDG_CACHE_HOME=/workspace/.cache \
--env FLASHINFER_WORKSPACE_DIR=/users/$USER_NAME/.cache/flashinfer \
--env TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torchinductor \
--env TRITON_CACHE_DIR=/workspace/.cache/triton \
--env VLLM_DISABLE_COMPILE=1 \
"$IMAGE" \
  python /workspace/tt2.py \
--model "$MODEL" \
--prompt "$PROMPT" \
--maxtokens "$MAXTOKENS" \
--temperature"$TEMPERATURE"

echo"===== JOB END ====="
date
```

### `-bind "$WORK_DIR:/workspace"`

Makes your host workspace visible inside the container as `/workspace`.

Why it matters:

- your generated Python file is stored there
- caches are stored there
- the container needs a writable location

### `-home "$HOME_DIR:/users/$USER_NAME"`

Sets a writable home directory inside the container.

Why it matters:

- some libraries still write to `$HOME`
- this avoids filling your real home directory quota
- it keeps cache behavior predictable

### `-env HF_HOME=/workspace/.cache/huggingface`

Controls where Hugging Face downloads model files.

### `-env XDG_CACHE_HOME=/workspace/.cache`

A general cache root used by many tools.

### `-env TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torchinductor`

Stores Torch compilation/cache artifacts.

### `-env TRITON_CACHE_DIR=/workspace/.cache/triton`

Stores Triton kernel caches.

### `-env FLASHINFER_WORKSPACE_DIR=/users/$USER_NAME/.cache/flashinfer`

Sets a writable FlashInfer workspace.

### `-env VLLM_DISABLE_COMPILE=1`

Useful for quick tests, because it can reduce compile overhead on short runs.

It may reduce performance, so it is best thought of as a **testing convenience**, not always the final production setting.

---

## 5 How to run it

Before submitting, make sure:

- you are in the GH environment
- the container image exists
- the script file is saved

Then submit:

```bash
sbatch run_vllm_inference.sbatch
```

To monitor the job:

```bash
squeue -u $USER --start
```

To open the output log (alternatively more/nano):

```bash
less vllm_infer_<jobid>.out
```

To follow it live:

```bash
tail-f vllm_infer_<jobid>.out
```

---

## 6 Practical notes on model choice

Your example uses:

```bash
MODEL="Qwen/Qwen2.5-32B-Instruct"
```

That is much larger than the earlier `facebook/opt-125m` smoke-test model.

So be careful:

- larger models take longer to download
- larger models use more memory
- first run may be slow due to cache warm-up

For a first validation, a smaller model is safer.

Once the workflow is proven, then move to the larger instruct model. Do not use larger models on the `ghtest`.  

---

## 7 What results to expect

If the job works correctly, the output log will usually show:

- job start lines
- hostname and date
- `nvidia-smi` output
- confirmation that the Python file was created
- the selected model name
- then a generated text response

The important section will look something like:

```
===== RESPONSE =====

Precision refers to how close repeated measurements are to each other, while accuracy refers to how close a measurement is to the true value...
```

That printed response is coming from this line in the Python file:

```
print(out.outputs[0].text)
```

So the Slurm output log is where your final answer appears.

---

## 8 Setting up multi-GPU inference

The previous example runs on a **single GPU**. If you want to use **multiple GPUs on the same node**, you must configure **both Slurm and vLLM** correctly. Simply requesting more GPUs in Slurm is not enough — the Python code must also be told how many GPUs to use.

This section explains the two places you must modify.

---

### A) Request multiple GPUs in the Slurm job

In the Slurm header, change the GPU request:

```bash
#SBATCH --gres=gpu:**2**
```

This asks Slurm to allocate **two GPUs on the node** for your job.

Example:

```bash
#SBATCH --job-name=vllm-infer
#SBATCH --partition=ghtest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:**2**
#SBATCH --time=00:15:00
```

Important notes:

- GPUs must be on **the same node** (which they will be when using `-nodes=1`).
- Multi-GPU jobs may take longer to schedule because they require more resources.

---

### B) Tell vLLM how many GPUs to use

Requesting GPUs from Slurm only gives the hardware.

You must also configure **vLLM’s tensor parallelism** so it knows to distribute the model across those GPUs.

Modify the Python code:

### Single-GPU version

```bash
llm=LLM(model=args.model, tensor_parallel_size=**1**)
```

### Multi-GPU version

```bash
llm=LLM(model=args.model, tensor_parallel_size=**2**)
```

`tensor_parallel_size` must match the number of GPUs requested from Slurm.

If these numbers do not match, vLLM will either:

- ignore the extra GPUs, or
- fail with a runtime error.

---

### C) Why multi-GPU is needed for larger models

Large models often cannot fit in the memory of a single GPU.

For example:

| Model | Approx size | GPUs typically needed |
| --- | --- | --- |
| OPT-125M | tiny | 1 |
| Llama-7B | small | 1 |
| Qwen-32B | large | 2–4 |
| 70B models | very large | 4+ |

Tensor parallelism allows vLLM to **split model weights across GPUs**, so each GPU holds part of the model.

---

### D) Practical advice

Multi-GPU jobs introduce a few practical considerations:

- **Scheduling may take longer** because more GPUs are required.
- **Models must support tensor parallelism**.
- **GPU memory must still be sufficient** even when split across devices.

For quick testing, it is often easier to start with **1 GPU and a smaller model**, then scale up once the workflow is confirmed.

---

## Pitfalls

### A) “ModuleNotFoundError: vllm”

You are not running inside the correct image, or the image doesn’t actually contain vLLM.

- Confirm you used `"$IMAGE"` everywhere.
- Confirm the base was the NVIDIA vLLM image during build.

### B) “Permission denied” / “Read-only file system”

A cache is writing somewhere that isn’t writable.

- Confirm `-bind "$WORK_DIR:/workspace"` is present.
- Confirm `HF_HOME`, `XDG_CACHE_HOME`, and the cache dirs are created (`mkdir -p` lines).
- Confirm you’re not accidentally writing to default `$HOME`.

### C) OOM / CUDA errors

Even small models should not OOM on GH, but CUDA errors can happen if the runtime libraries aren’t being injected.

- Confirm `apptainer exec --nv` is used.
- Check `nvidia-smi` output at the top of the logs.

### D) It spends the whole job “doing nothing”

Often it is downloading weights on first run.

- Check the log with `tail -f vllm_infer_<jobid>.out`
- Ensure HF cache is on `/nobackup` so the second run is fast.

---
