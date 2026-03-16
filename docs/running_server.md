# Run a vLLM OpenAI-compatible server from your Apptainer image on Bede GH nodes

This guide shows how to start a **vLLM server** from an existing `.sif` image using Slurm on Bede GH nodes. It keeps things honest: the server runs **on the compute node**, not on your laptop, and reaching it usually requires **being on the same node** (or using SSH tunnelling / port forwarding via the HPC login path, if allowed).

For first tests, we’ll use the `ghtest` partition (short jobs) and a small public model.

---

## 1  Why you should run the server via Slurm (not interactively)

A vLLM server needs:

- a GPU allocation (`-gres=gpu:1`)
- a stable runtime window (`-time=...`)
- consistent environment and filesystem write locations

On HPC, **compute nodes are the good place** to do this. Slurm ensures:

- you actually get a GPU
- your server isn’t killed because you ran it somewhere you shouldn’t
- logs go to files you can inspect (`.out` / `.err`).

Jobs targeting the`gh` or `ghtest`partitions are usually submitted from a `ghlogin`

session.

Alternatively, they can be submitted from the regular login nodes using `ghbatc`or `ghrun`.

---

## 2) Must-have: submit from `ghlogin`

Before using `ghtest`/`gh`, you should be in the GH environment login session:

```bash
ghlogin -A <project_ID>
```

From there, you submit the server job with `sbatch`.

---

## 3) The Slurm script: `run-vllm-server.sbatch`

Here is your script, followed by an explanation of what each part is doing and why.

```bash
#!/bin/bash
#SBATCH --job-name=vllm-server
#SBATCH --partition=ghtest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH -A <project_ID>
#SBATCH --output=vllm_server_%j.out
#SBATCH --error=vllm_server_%j.err

set-euo pipefail

echo"===== JOB START ====="
echo"Host:$(hostname)"
echo"Date:$(date)"
echo"GPU:"
nvidia-smi

PROJ=""
USER=""
IMAGE="vllm-ag2-26.01.1.sif"
if [[ !-f"$IMAGE" ]];then
echo"ERROR: container image $IMAGE not found"
exit1
fi

# --------- writable cache/work dirs on /nobackup ---------
BASE="/nobackup/projects/$PROJ/$USER/modules/vllm26_server"
HOME_DIR="$BASE/home"# will map to /users/user in container
WORK_DIR="$BASE/workspace"# will map to /workspace in container

mkdir -p "$HOME_DIR" "$WORK_DIR"
chmod -R a+rwx "$BASE"

# Pick a port
PORT=8080

# Model to serve (small public model for testing)
MODEL="facebook/opt-125m"

echo "===== STARTING vLLM SERVER ====="
echo "Model:$MODEL"
echo "Port:$PORT"
echo "Logs:  vllm_server_${SLURM_JOB_ID}.out / .err"
echo
echo "After it starts, test from the SAME NODE with:"
echo "  curl http://0.0.0.0:${PORT}/v1/models"
echo

apptainer exec--nv \
--bind "$WORK_DIR:/workspace" \
--home "$HOME_DIR:/users/$USER" \
--env HF_HOME=/workspace/.cache/huggingface \
--env XDG_CACHE_HOME=/workspace/.cache \
--env FLASHINFER_WORKSPACE_DIR=/users/$USER/.cache/flashinfer \
--env TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torchinductor \
--env TRITON_CACHE_DIR=/workspace/.cache/triton \
--env TERM=dumb \
"$IMAGE" bash-lc "
set -eux
mkdir -p /workspace/.cache/huggingface
mkdir -p /workspace/.cache/torchinductor
mkdir -p /workspace/.cache/triton
mkdir -p /users/$USER/.cache/flashinfer

# Start OpenAI-compatible server
# --host 0.0.0.0 binds to all interfaces on the node (needed for tunneling)
vllm serve \"$MODEL\" \
  --host 0.0.0.0 \
  --port${PORT} \
  --dtype float16 \
  --tensor-parallel-size 1
"

echo"===== JOB END ====="
date
```

---

## 4) What the Slurm directives mean (and why they’re chosen)

- `-partition=ghtest`
    
    Short test partition: good for verifying the server starts and can load a small model.
    
- `-gres=gpu:1`
    
    vLLM needs a GPU. Without this, you’ll either crash or run on CPU (which is not what you want here).
    
- `-time=00:15:00`
    
    A sanity check window. If the model downloads, loads, and serves within 15 minutes, you know the plumbing works. For real models, you’ll increase this.
    
- `-output` / `-error` with `%j`
    
    Separate stdout/stderr logs per job ID. Very helpful when debugging why the server didn’t start.
    

---

## 5) Why you need `-home`, `-bind`, and the environment variables

### `-bind "$WORK_DIR:/workspace"`

Apptainer containers are often effectively read-only at runtime (especially the SIF filesystem).

Binding a host directory to `/workspace` gives you a **writable location inside the container** for:

- Hugging Face model downloads
- compilation caches (Triton / TorchInductor)
- temporary artifacts

And you deliberately put it on `/nobackup` for performance and to avoid `$HOME` quotas.

### `-home "$HOME_DIR:/users/$USER"`

This is easy to misunderstand, so here’s the honest version:

- Many tools assume `$HOME` exists and is writable.
- On HPC, your real `$HOME` might be quota-limited or slow.
- Some libraries cache under `~/.cache` by default.

By setting `--home "$HOME_DIR:/users/$USER"` you are:

- telling Apptainer: “treat this directory on `/nobackup` as the user’s home inside the container”
- ensuring anything that writes to `~/.cache`, `~/.config`, etc. goes to a **writable, fast** place

**Why map it specifically to `/userers/``?**

Because that matches the expected home path used inside your environment / Bede conventions. The exact container-side home path doesn’t have to be `/users/$USER`, but you must be consistent: the caches you place under `/users/$USER/...` should align with where `$HOME` points.

### Why all the `-env ...` lines exist

These force caches to your writable bind-mounted directories instead of random defaults:

- `HF_HOME=/workspace/.cache/huggingface`
    
    Where Hugging Face stores model weights and tokenizer files.
    
- `XDG_CACHE_HOME=/workspace/.cache`
    
    A broad “catch-all” cache location used by many Linux tools.
    
- `TRITON_CACHE_DIR=/workspace/.cache/triton`
    
    Triton (used heavily by vLLM) compiles GPU kernels and caches them. If this points somewhere unwritable, you can get failures or repeated recompiles.
    
- `TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torchinductor`
    
    TorchInductor compilation artifacts, similar story.
    
- `FLASHINFER_WORKSPACE_DIR=/users/$USER/.cache/flashinfer`
    
    If FlashInfer is used (or optionally used) it may want a workspace directory. You’re pinning it to a known writable cache area.
    
- `TERM=dumb`
    
    Mostly to keep logs clean and avoid weird terminal capability issues in non-interactive batch jobs.
    

These caches prevent repeated GPU kernel recompilation between runs.

### The `bash -lc " ... "` wrapper

This:

- runs a login-like shell command string
- lets you use shell variables and quoting in a single block
- ensures the command sequence runs inside the container in one session

Inside it, you create cache directories explicitly (`mkdir -p ...`) to avoid “directory missing” issues during first run.

---

## 6) Binding to `0.0.0.0` and what it actually means

You run:

```bash
vllm serve "$MODEL" --host0.0.0.0 --port 8000
```

This **does not** make the server globally accessible on the internet.

What it actually does:

- binds the service to all interfaces *on that compute node*
- which is usually required if you want to reach it via SSH tunnelling or from another process on the same node

Access rules still depend on:

- Bede firewall / node network isolation
- whether compute nodes accept incoming connections from your login session
- whether port forwarding is permitted

So the safe statement is:

> `--host 0.0.0.0` allows the server to accept connections on the compute node interfaces, which is useful for tunnelling. You still need an HPC-appropriate access method to reach it.
> 

---

## 7) How to run it

From `ghlogin`, go to the directory containing the script and image:

```bash
cd /nobackup/projects/$PROJ/$USER/containers
# or wherever you keep it sbatch run-vllm-server.sbatch
```

Monitor:

```bash
squeue -u $USER
```

When it starts, open the output log:

```bash
more vllm_server_<jobid>.out
```

You will see the hostname. That hostname is the node where the server is running.

---

## Pitfalls

- **Model download time:** even “small” models may need time on first run.
    
    Fix: keep caches on `/nobackup` so you only download once.
    
- **Cache not writable:** most common vLLM container failure on HPC.
    
    Fix: your `--bind`, `--home`, and cache env vars are exactly what prevents this.
    
- **Networking assumptions:** don’t assume you can reach compute nodes directly from your laptop.
    
    Fix: plan for “same-node test” or tunnelling if policy allows.
    
- **`ghtest` time too short for big models:** 15 minutes is great for smoke tests, not for real workloads.
    
    Fix: move to `--partition=gh` and increase time when you serve real models.
    

---

## 8) Testing the server

A Slurm batch job runs your server on a **compute node**. Your original shell (and your laptop) are usually **not on that node**, so `curl 0.0.0.0:PORT` will only work **from the compute node itself**.

The clean workflow is:

1. submit the job
2. find which node it landed on
3. in a **second terminal**, connect to that node (if permitted) and test locally
4. optionally set up SSH port forwarding to reach it from your laptop

### Step 1 — Submit the job

From `ghlogin`:

```bash
sbatch run-vllm-server.sbatch
```

### Step 2 — Find the job’s node and job ID

```bash
squeue -u $USER --start
```

Look for your job and note:

- the **JOBID**
- the **NODELIST** (e.g., `gg001` or similar)

If the job is already running, you can also do:

```bash
scontrol show job <JOBID> |grep -E "JobId=|NodeList="
```

---

## 9) Testing the server from a second terminal

### Option A (most direct): SSH onto the compute node and curl localhost

This works **only if** Bede allows SSH from the login environment to compute nodes (many HPC systems do; some restrict it).

1. Open a **second terminal**
2. Connect to Bede and enter the GH environment:

```bash
ghlogin
```

1. SSH to the node where the job is running (replace `NODE`):

```
ssh NODE
```

1. Now you are *on the same node* as the server, so this will work:

```bash
curl http://0.0.0.0:8000/v1/models
```

If the server is alive you should get a JSON response (models list), and `/health` is another simple check:

```bash
curl -s http://0.0.0.0:8000/health
```

**Important:** if SSH to compute nodes is blocked on your setup, use Option B below.

---

## 10) Opening the server log files

Your Slurm script writes logs here (in the directory you submitted from):

- stdout: `vllm_server_<JOBID>.out`
- stderr: `vllm_server_<JOBID>.err`

### Quick ways to view logs

- Scroll through the log:

```bash
less vllm_server_<JOBID>.out
```

- Follow the log live (best while waiting for model download / startup):

```bash
tail-f vllm_server_<JOBID>.out
```

- If something fails, check stderr:

```bash
less vllm_server_<JOBID>.err
```

### What “working” logs look like

When the server is healthy, you typically see lines like the ones you posted. Key “green flag” indicators:

- Engine initialization / warmup completed:

```bash
(EngineCore_DP0 pid=525842) INFO ... init engine (profile, create kv cache, warmup model) took 5.40 seconds
```

- API server announces supported tasks and the bind address:

```bash
(APIServer pid=525681) INFO ... Supported tasks: ['generate']
(APIServer pid=525681) INFO ... Starting vLLM API server 0 on http://0.0.0.0:8080
```

- Routes are listed (this is a strong sign it’s actually serving HTTP):

```bash
(APIServer pid=525681) INFO ... Available routes are:
(APIServer pid=525681) INFO ... Route: /v1/models, Methods: GET
(APIServer pid=525681) INFO ... Route: /health, Methods: GET
(APIServer pid=525681) INFO ... Route: /v1/chat/completions, Methods: POST
...
```

- Submited and executed prompts:

```bash
(APIServer pid=525681) INFO:     127.0.0.1:46118 - "GET /v1/models HTTP/1.1" 200 OK
(APIServer pid=525681) INFO:     127.0.0.1:36340 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=525681) INFO 03-02 07:37:37 [loggers.py:248] Engine 000: Avg prompt throughput: 1.3 tokens/s, Avg generation throughput: 1.3 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
(APIServer pid=525681) INFO 03-02 07:37:47 [loggers.py:248] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
```

**What to look for in your logs specifically**

- The line `Starting vLLM API server ... http://0.0.0.0:<PORT>` should match the `PORT` you set in the script.
- If you don’t see it, the server likely didn’t reach the “listening” stage (often model download, OOM, or cache permission issues).

---
