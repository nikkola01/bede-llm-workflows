# Running llama.cpp on Bede

- how to enter the right environment (Grace Hopper side, not the legacy Power9 side),
- how to build llama.cpp with CUDA for H100,
- how to download a model,
- how to submit a GPU job to the `gh` / `ghtest` partition,

## 1. Get into the Grace Hopper environment

From a normal Bede login node:

```bash
ghlogin -A <your_project_code>
```

This gives you an interactive shell on a Grace CPU-only login node (aarch64, Rocky 9).

No GPU there, but that’s okay — we’ll compile here with CUDA stub libs.

## 2. Load CUDA on ghlogin (for compilation)

On that ghlogin shell:

```bash
module avail            # just to see what's there
module load cuda/12.6.2 
```

Bede’s docs show `module load cuda/12.6.2` as an example for GH, plus an `LD_LIBRARY_PATH` trick to expose stub driver libs so you can link against CUDA with no GPU present. [Bede Documentation](https://bede-documentation.readthedocs.io/en/latest/usage/index.html?utm_source=chatgpt.com)

Run:

```bash
export LD_LIBRARY_PATH=/usr/lib64:$CUDA_HOME/lib64/stubs:$LD_LIBRARY_PATH
```

Why this matters:

- On ghlogin there’s no physical GPU, so CUDA gives you “stub” libcuda.so.
- Setting `LD_LIBRARY_PATH` like that lets CMake link llama.cpp against CUDA/cuBLAS successfully, even though you’re on a CPU-only login node. This is exactly how Bede expects you to prep GPU workloads.

## 3. Prepare a working directory on GH side

Still in ghlogin:

```bash
mkdir -p ~/bede-llama
cd ~/bede-llama

# get llama.cpp
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```

We’re now sitting in `~/bede-llama/llama.cpp` on an aarch64 Grace node.

## 4. Configure and build llama.cpp for the H100 on Bede

We want:

- cuBLAS on (GPU acceleration),
- correct arch flags for Hopper (sm_90),
- no tests to avoid long compile times,
- low-ish parallelism so we don’t blow memory in the interactive login session.

Do this:

```bash
cd ..cmake -B build \
  -DGGML_CUBLAS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90

cmake --build build -j4
```

Why:

- `GGML_CUBLAS=ON`: tells llama.cpp to build GPU support via CUDA/cuBLAS.
- `CMAKE_CUDA_ARCHITECTURES=90`: Hopper/H100 compute capability 9.0, exactly what Bede docs require for GH200 jobs. [Bede Documentation](https://bede-documentation.readthedocs.io/en/latest/usage/index.html?utm_source=chatgpt.com)
- `DLLAMA_BUILD_TESTS=OFF`: we don’t need a thousand unit tests on the login node.
- `DLLAMA_BUILD_EXAMPLES=ON`: this gives you `llama-cli`, `llama-server`, etc. You could set this to `OFF` too if you just want `llama-cli` fastest, but usually you want at least the CLI.

If RAM is tight or the job is getting OOM-killed while compiling, you can drop `-j4` to `-j2` or `-j1`.

When this finishes, you should have binaries like:

```bash
ls build/bin
# expect: llama-cli, llama-server, etc.
```

Now you’ve got an aarch64 + CUDA + sm_90 build of llama.cpp that should run on the GH compute nodes with GPUs.

## 5. Download a small model to test with later

We now grab a tiny quantized model and store it in your home, so both the login shell and the GPU job can see it.

We now grab a tiny quantized model and store it in your home, so both the login shell and the GPU job can see it.

```bash
#doen't work!
pip3 install --user "huggingface_hub[cli]"
mkdir -p ~/models

huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  --include "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
  --local-dir ~/models \
  --local-dir-use-symlinks False
```

We'll attempt a direct download with `wget` so we can see if network is the blocker or the CLI.

Try:

```bash
mkdir -p ~/models
cd ~/models
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
ls -lh
```

Check:

```bash
ls -lh ~/models
```

You should now have `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`.

***** make a  hugging face - can we download just by writing the name of the module

200B

## 6. Make a Slurm job script to actually run on a GH GPU node

- (Optional) Check Specs
    
    Do this from `gg001.bede`(inside ghlogin):
    
    ```bash
    sinfo -o "%P %N %c %m %G"
    sinfo -p gh -o "%P %N %c %m %G"
    sinfo -p ghtest -o "%P %N %c %m %G"
    ```
    
    We can see:
    
    - partition name (`gh` vs `ghtest`),
    - CPUs per node,
    - memory per node,
    - GPU GRES string,
    

Now create a file in `~/bede-llama` called `run_llama_gh.sbatch`:

```bash
nano ~/bede-llama/run_llama_gh.sbatch
```

Write this:

```bash
#!/bin/bash
#SBATCH -J llama-gh
#SBATCH --partition=gh
##SBATCH --ntasks=1
##SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:05:00
#SBATCH -A bddur53
#SBATCH -o llama-%j.out

module purge
module load cuda/12.6.2

export LD_LIBRARY_PATH=/usr/lib64:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

MODEL_DIR="${HOME}/models"
MODEL_FILE="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
LLAMA_DIR="${HOME}/bede-llama/llama.cpp"

echo "[INFO] model: ${MODEL_DIR}/${MODEL_FILE}"
echo "[INFO] running llama-cli on GH GPU node..."

${LLAMA_DIR}/build/bin/llama-cli \
  -m ${MODEL_DIR}/${MODEL_FILE} \
  -p "Hello from Bede Grace Hopper H100." \
  --n-gpu-layers 100 \
  -n 64
```

Exit and Safe (Ctrl+X)
The double semicolon ## is a comment. It seems that Bede doesn’t like too much specificity.

## 7. Submit it to actually run on GPU

From `~/bede-llama`:

```bash
cd ~/bede-llama
sbatch run_llama_gh.sbatch
```

Then check your queue:

```bash
squeue -u $USER
```

To cancel a job:

```bash
scancel <job-ID>
```

When it finishes, read the output:

```bash
cat llama-*.out
```

First result:

```bash
nano llama-937608.out
```

```bash
llama_context: graph nodes  = 689
llama_context: graph splits = 1
common_init_from_params: added </s> logit bias = -inf
common_init_from_params: setting dry_penalty_last_n to ctx_size = 4096
common_init_from_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)
main: llama threadpool init, n_threads = 72
main: model was trained on only 2048 context tokens (4096 specified)
main: chat template is available, enabling conversation mode (disable it with -no-cnv)
*** User-specified prompt will pre-start conversation, did you mean to set --system-prompt (-sys) instead?
main: chat template example:
<|system|>
You are a helpful assistant<|user|>
Hello<|assistant|>
Hi there<|user|>
How are you?<|assistant|>

system_info: n_threads = 72 (n_threads_batch = 72) / 72 | CPU : NEON = 1 | ARM_FMA = 1 | FP16_VA = 1 | MATMUL_INT8 = 1 | SVE = 1 | DOTPROD = 1 | SVE_CNT = 16 | OPENMP = 1 | REPACK>

main: interactive mode on.
sampler seed: 2761594732
sampler params:
        repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
        dry_multiplier = 0.000, dry_base = 1.750, dry_allowed_length = 2, dry_penalty_last_n = 4096
        top_k = 40, top_p = 0.950, min_p = 0.050, xtc_probability = 0.000, xtc_threshold = 0.100, typical_p = 1.000, top_n_sigma = -1.000, temp = 0.800
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
sampler chain: logits -> logit-bias -> penalties -> dry -> top-n-sigma -> top-k -> typical -> top-p -> min-p -> xtc -> temp-ext -> dist
generate: n_ctx = 4096, n_batch = 2048, n_predict = 64, n_keep = 1

== Running in interactive mode. ==
 - Press Ctrl+C to interject at any time.
 - Press Return to return control to the AI.
 - To return control without starting a new line, end your input with '/'.
 - If you want to submit another line, end your input with '\'.
 - Not using system message. To change it, set a different value via -sys PROMPT

 <|user|>
Hello from Bede Grace Hopper H100.<|assistant|>
How can I find out if a specific hotel or resort has an in-room wireless internet connection in Bede, ND?

> EOF by user

llama_perf_sampler_print:    sampling time =	   0.85 ms /    52 runs   (    0.02 ms per token, 61104.58 tokens per second)
llama_perf_context_print:        load time =    1507.13 ms
llama_perf_context_print: prompt eval time =	  19.03 ms /    24 tokens (    0.79 ms per token,  1261.30 tokens per second)
llama_perf_context_print:        eval time =     158.11 ms /    27 runs   (    5.86 ms per token,   170.77 tokens per second)
llama_perf_context_print:	total time =     183.53 ms /    51 tokens
llama_perf_context_print:    graphs reused =         26
llama_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
llama_memory_breakdown_print: |   - Host               |                  794 =   636 +      88 +      70                |
```

# Create Llama.cpp surver

## Create `run_llama_server.sbatch`

Create this file:

```bash
nano ~/bede-llama/run_llama_server.sbatch
```

Paste **this exact script** (clean, correct, and GH-friendly):

```bash
#!/bin/bash
#SBATCH -J llama-server
#SBATCH --partition=gh
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=02:00:00
#SBATCH -A bddur53
#SBATCH -o llama-server-%j.out

module purge
module load gcc/12.2
module load cuda/12.6.2

# expose CUDA stub libs if needed
export LD_LIBRARY_PATH=/usr/lib64:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

MODEL_DIR="${HOME}/models"
MODEL_FILE="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
LLAMA_DIR="${HOME}/bede-llama/llama.cpp"

echo "[INFO] starting llama-server on GH GPU node..."
echo "[INFO] model: $MODEL_DIR/$MODEL_FILE"

# Start llama-server on port 8080
# (--host 0.0.0.0 allows access via ssh port forwarding)
${LLAMA_DIR}/build/bin/llama-server \
  -m ${MODEL_DIR}/${MODEL_FILE} \
  --n-gpu-layers 100 \
  --port 8080 \
  --host 0.0.0.0
```

Save with **Ctrl+X → Y → Enter**

## Submit the server job

In the GH login node (`ghlogin -A bddur53`):

```bash
cd ~/bede-llama
sbatch run_llama_server.sbatch
```

Watch queue:

```bash
squeue -u $USER
```

When it’s running, check output:

```bash
tail -f llama-server-*.out
```

You should see:

```
[INFO] starting llama-server on GH GPU node...
llama-server listening on 0.0.0.0:8080
```

## Connect to the server (SSH port forwarding)

From your local machine:

```bash
ssh -L 8080:<gpu-node-name>:8080 username@login.bede.ac.uk
```

Where `<gpu-node-name>` is shown in `squeue` (e.g. `gh001`).

Then open:

```
http://localhost:8080
```

or query via curl:

```bash
curl -X POST http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello from client"}'
```

```bash
llama_sbatch_templates/
├── 01_prompt.sbatch
├── 02_prompt_file.sbatch
├── 03_chat.sbatch
├── 04_server.sbatch
└── 05_benchmark.sbatch
├── 06_multigpu_server.sbatch (hopefully)
└── [README.md](http://readme.md/)
```

01_prompt.sbatch

```bash
#!/bin/bash
#SBATCH --job-name=llama-prompt
#SBATCH --partition=gh
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:05:00
#SBATCH -A <project_ID>
#SBATCH -o llama-prompt-%j.out
#SBATCH -e llama-prompt-%j.err

set -euo pipefail

LLAMA_BIN=/nobackup/projects/<project_ID>/<user_ID>/module_homes/aarch64/llama.cpp/build/bin/llama-cli
MODEL=/nobackup/projects/<project_ID>/modules/Qwen2.5-7B-Instruct.Q4_K_M.gguf

PROMPT="Explain why chia seeds are considered healthy."

echo "[INFO] Running prompt on $(hostname)"
echo "[INFO] Model: $MODEL"

$LLAMA_BIN \
  -m "$MODEL" \
  -p "$PROMPT" \
  --n-predict 256 \
  --ctx-size 4096 \
  --gpu-layers 999
```

02_prompt_file.sbatch

```bash
#!/bin/bash
#SBATCH --job-name=llama-file
#SBATCH --partition=gh
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:10:00
#SBATCH -A <project_ID>
#SBATCH -o llama-file-%j.out
#SBATCH -e llama-file-%j.err

set -euo pipefail

LLAMA_BIN=/nobackup/projects/<project_ID>/llama.cpp/build/bin/llama-cli
MODEL=/nobackup/projects/<project_ID>/models/Qwen2.5-7B-Instruct.Q4_K_M.gguf
PROMPT_FILE=prompt.txt

$LLAMA_BIN \
  -m "$MODEL" \
  -f "$PROMPT_FILE" \
  --n-predict 512 \
  --ctx-size 4096 \
  --gpu-layers 999

```

03_chat.sbatch

```bash
#!/bin/bash
#SBATCH --job-name=llama-chat
#SBATCH --partition=gh
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=01:00:00
#SBATCH -A <project_ID>
#SBATCH --pty bash

LLAMA_BIN=/nobackup/projects/<project_ID>/llama.cpp/build/bin/llama-cli
MODEL=/nobackup/projects/<project_ID>/models/Qwen2.5-7B-Instruct.Q4_K_M.gguf

$LLAMA_BIN \
  -m "$MODEL" \
  --interactive \
  --ctx-size 4096 \
  --gpu-layers 999

```

 04_server.sbatch

```bash
#!/bin/bash
#SBATCH --job-name=llama-server
#SBATCH --partition=gh
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=04:00:00
#SBATCH -A <project_ID>
#SBATCH -o llama-server-%j.out
#SBATCH -e llama-server-%j.err

set -euo pipefail

LLAMA_SERVER=/nobackup/projects/<project_ID>/llama.cpp/build/bin/llama-server
MODEL=/nobackup/projects/<project_ID>/models/Qwen2.5-7B-Instruct.Q4_K_M.gguf
PORT=8080

echo "[INFO] Starting llama.cpp server on $(hostname):$PORT"

$LLAMA_SERVER \
  -m "$MODEL" \
  --port $PORT \
  --ctx-size 4096 \
  --gpu-layers 999 \
  --threads 72

```

```bash
curl http://NODE_NAME:8080/completion

```

05_benchmark.sbatch

```bash
#!/bin/bash
#SBATCH --job-name=llama-bench
#SBATCH --partition=gh
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:05:00
#SBATCH -A <project_ID>
#SBATCH -o llama-bench-%j.out

LLAMA_BIN=/nobackup/projects/<project_ID>/llama.cpp/build/bin/llama-bench
MODEL=/nobackup/projects/<project_ID>/models/Qwen2.5-7B-Instruct.Q4_K_M.gguf

$LLAMA_BIN \
  -m "$MODEL" \
  --ctx-size 4096 \
  --n-prompt 512 \
  --n-gen 256

```
