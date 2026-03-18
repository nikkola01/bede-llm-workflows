# Build a vLLM Apptainer container on Bede GH nodes (using NVIDIA’s vLLM base image)

### Why this approach

On Bede’s `gh` nodes (Grace Hopper / aarch64 + Hopper GPU), **building vLLM from scratch in a “normal” base image is a trap**:

- You can hit **CUDA / compiler / driver mismatches**
- You can hit **binary wheel incompatibilities on aarch64**
- You can end up with a Franken-environment where Torch, CUDA libs, and vLLM don’t agree

The simplest reliable path is to start from **NVIDIA’s official vLLM container image** (`nvcr.io/nvidia/vllm:<tag>`), where **vLLM + matching CUDA/Torch** are already integrated. (This is especially relevant now that NVIDIA has maintained vLLM images in their NGC ecosystem since January 2026.)

**Bottom line:** use the NVIDIA vLLM image (or an equally appropriate vendor-provided vLLM base). If you try to pip-install vLLM on a random CUDA image, expect pain. (I did :)

---
## 1) Definition file: extend NVIDIA’s vLLM base and add your extras (AG2/AutoGen)

### Your definition file

```
Bootstrap: docker
From: nvcr.io/nvidia/vllm:26.01-py3

%post
    set -eux

    # (Optional) keep pip tooling current, but don't touch torch/vllm
    python3 -m pip install --upgrade pip setuptools wheel

    # Use NVIDIA’s constraints file if present (prevents dependency drift)
    CONSTRAINT=""
    if [ -f /etc/pip/constraint.txt ]; then
        CONSTRAINT="-c /etc/pip/constraint.txt"
        echo "Using pip constraints at /etc/pip/constraint.txt"
    fi

    # AG2 / AutoGen deps you previously hit
    python3 -m pip install ${CONSTRAINT} \
        diskcache \
        cbor2

    # Install AG2 and/or AutoGen
    python3 -m pip install ${CONSTRAINT} \
        ag2 \
        autogen

    # Create a writable workspace inside container
    mkdir -p /workspace

%environment
    export HF_HOME=/workspace/.cache/huggingface
    export TOKENIZERS_PARALLELISM=false

%runscript
    exec "$@"
```

### What each section does (and why)

### `Bootstrap: docker` / `From: nvcr.io/nvidia/vllm:26.01-py3`

- Uses a Docker image or later versions as the base layer.  [pytorch-26-01](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-01.html)
- **This is the most important line in the whole guide**: you’re inheriting NVIDIA’s known-good stack where vLLM is already aligned with CUDA + Torch versions.

### `%post`

Runs at build time inside the container filesystem.

- `set -eux`
    - `e`: fail on error
    - `u`: fail on unset vars
    - `x`: print commands as they execute (super helpful for build logs)
- `pip install --upgrade pip setuptools wheel`
    
    Keeps packaging tools modern.
    
    **Note:** You explicitly do *not* touch torch/vllm here — good. Touching Torch is how you create ABI mismatches.
    
- Constraints handling (`/etc/pip/constraint.txt`)
    
    If NVIDIA provides a constraints file, using it helps prevent “dependency drift” where pip upgrades something that breaks the pre-built stack.
    
- Install your extras (`diskcache`, `cbor2`, `ag2`, `autogen`)
    
    These are your application-layer dependencies. Installing them on top of the base image is the right pattern.
    
- `mkdir -p /workspace`
    
    Creates a standard location you can bind/mount over, and also a place for caches.
    

### `%environment`

Environment vars that are set whenever you run the container.

- `HF_HOME=/workspace/.cache/huggingface`
    
    Hugging Face cache goes somewhere you can make writable (and ideally bind to a project path).
    
- `TOKENIZERS_PARALLELISM=false`
    
    Avoids noisy tokenizer parallelism warnings and can reduce CPU oversubscription in HPC contexts.
    

### `%runscript`

Defines what happens when you run the container directly.

- `exec "$@"` means: “whatever command the user passes, run it as-is”.
- This is the most flexible pattern for HPC usage.

---

## 2) How you typically run it on Bede (pattern)

After building `vllm-all-dep3.sif`, you usually want:

- GPU access
- writable cache/workspace
- your code mounted in

Typical pattern (example):

```
apptainer exec--nv \
--bind /nobackup/projects/$PROJ/$USER:/workspace \
  vllm-all-dep3.sif \
  python-c"import vllm; print(vllm.__version__)"
```

(Adjust binds to your preferred project paths.)

---

### Pitfalls and how to avoid them

###  1 - “I’ll just pip install vllm in a CUDA image”

This is the #1 failure mode.

- vLLM depends on tightly-coupled CUDA + Torch + compiler toolchain pieces.
- On GH (aarch64), binary availability can be different than x86.
    
**Fix:** start from NVIDIA’s vLLM base (`nvcr.io/nvidia/vllm:<tag>`) or an equivalent purpose-built image.
    

### 2 - Building on the wrong architecture

If you build on x86 and try to run on GH (ARM), you can end up with incompatible artifacts or assumptions.

**Fix:** build on Bede in the `gh` environment.

### 3 - Caches in `$HOME` (quota + performance issues)

Apptainer build cache and tmp can explode in size.

**Fix:** always point `APPTAINER_CACHEDIR` and `APPTAINER_TMPDIR` at `/nobackup/...`.

### 4 - Accidentally upgrading Torch/CUDA dependencies

A casual `pip install -U torch` or dependency resolving without constraints can silently replace core components.

**Fix:** don’t touch torch/vllm; use constraints if provided.

### 5 - Hugging Face cache not writable

If `HF_HOME` points somewhere read-only inside the SIF, model downloads fail.

**Fix:** set `HF_HOME` to a writable path and bind it to storage (`/nobackup/...`).

### 6 - Forgetting `-nv`

Without `--nv`, you’ll have no GPU libs inside the container runtime.

**Fix:** always use `apptainer exec --nv ...` for GPU jobs.

---

## 3) Why you should build with a Slurm script (and what each line is doing)

On Bede (and most HPC systems), you *can* sometimes run `apptainer build ...` interactively — but you often **shouldn’t**. A build can be CPU + RAM heavy, create lots of temporary files, and take long enough that interactive sessions (or login-node policies) become a problem. Running container builds on login nodes can degrade service for all users and may violate cluster policy.

Using a Slurm batch script gives you:

- **A compute node** (not a login node), which is the “approved” place to do heavy work
- **Predictable resources** (memory/time), so the build doesn’t get killed mid-way
- **Clean logging** (stdout/stderr captured to a file)
- **Reproducibility** (you can re-run the same build job later)

### The build script: `build-vllm-sif.sbatch`

```
#SBATCH --job-name=build-vllm-sif
#SBATCH --partition=ghtest
#SBATCH --time=00:45:00
#SBATCH -A <project_id>
#SBATCH --mem=16G
#SBATCH -o build-vllm-sif-%j.out

set-euo

echo"[INFO] Building SIF on$(hostname) at$(date)"

export PROJ=bddur53
export APPTAINER_CACHEDIR=/nobackup/projects/$PROJ/$USER/apptainer-cache
export APPTAINER_TMPDIR=/nobackup/projects/$PROJ/$USER/apptainer-tmp

mkdir -p "$APPTAINER_CACHEDIR""$APPTAINER_TMPDIR"

cd /nobackup/projects/$PROJ/$USER/containers

apptainer build vllm-all-dep3.sif vllm-all-dep.def

echo "[INFO] Build finished at$(date)"
```

### Slurm directives (the `#SBATCH ...` lines)

- `#SBATCH --job-name=build-vllm-sif`
    
    Names the job in `squeue` / `sacct` so you can find it easily.
    
- `#SBATCH --partition=gh`
    
    Runs on the **GH partition**. This matters because you’re ultimately targeting the GH environment (Grace/Hopper). Even if you’re not using the GPU for the build, building “in-family” reduces surprises (filesystem paths, platform assumptions, etc.).
    
- `#SBATCH --time=01:00:00`
    
    Sets a hard wall-time limit. Builds can sometimes hang on downloads or dependency resolution; this prevents runaway jobs.
    
- `#SBATCH -A bddur53`
    
    Charges the job to your project allocation.
    
- `#SBATCH --mem=32G`
    
    Reserves memory. Apptainer builds can expand layers, unpack filesystems, and compile Python wheels—this can spike RAM. If you under-allocate, the job may die with OOM.
    
- `#SBATCH -o build-vllm-sif-%j.out`
    
    Writes logs to a file, where `%j` becomes the job ID. This makes it easy to review failures without scrolling terminal history.

### Apptainer cache + temporary directories (critical on Bede)

- `export PROJ=<project_id>`
    
    Convenience variable so you don’t repeat the project string.
    
- `export APPTAINER_CACHEDIR=/nobackup/projects/$PROJ/$USER/apptainer-cache`
    
    Apptainer caches downloaded layers and intermediate build content. Putting this on `/nobackup`:
    
    - avoids home directory quotas
    - improves performance
    - lets future builds reuse cached layers
    
    Without this, Apptainer may fill your home directory quota during builds.
    
- `export APPTAINER_TMPDIR=/nobackup/projects/$PROJ/$USER/apptainer-tmp`
    
    Builds create large temporary files (unpacked layers, staging dirs). `/nobackup` is the right place for that.
    
- `mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"`
    
    Ensures those directories exist before the build starts. Quoted variables prevent whitespace/path issues.
    

### Build location + command

- `cd /nobackup/projects/$PROJ/$USER/containers`
    
    Keeps the `.sif` output (and the `.def` file) in project storage, not `$HOME`.
    
    Also avoids writing large artifacts to slower or quota-limited areas. 
    
- `apptainer build vllm-all-dep3.sif vllm-all-dep.def`
    
    Builds the SIF from the definition file:
    
    - output: `vllm-all-dep3.sif`
    - recipe: `vllm-all-dep.def`
    
    This is where Apptainer pulls the base Docker layers, runs `%post`, and assembles a sealed runtime image.
    
- `echo "[INFO] Build finished at $(date)"`
    
    Gives you a clear “done” marker in the log so you can quickly see whether the build completed or died mid-way.
    

### How to run this script

```bash
sbatch build-vllm-sif.sbatch
```

---

### Pitfalls (build-specific)

- **Building on a login node**: may be slow, may violate policy, may get killed.
    
    **Fix:**  Use Slurm batch.
    
- **Cache/tmp in `$HOME`**: quotas + terrible performance.
    
    **Fix:**  Use `/nobackup/...` via `APPTAINER_CACHEDIR` and `APPTAINER_TMPDIR`.
    
- **Not enough memory**: build dies during unpacking or wheel compilation.
    
    **Fix:** 32GB is a good starting point; bump if you see OOM in logs.
    
- **Disk space**: a single image + cache can be tens of GB.
    
    **Fix:** Keep builds, caches, and model caches on `/nobackup`.
    

## 4) Submitting the build job

Before running the build job, you **must first connect to the GH environment using `ghlogin`**. On Bede, the GH partition is part of a separate Grace/Hopper environment, and Slurm submissions targeting that partition should be made from a **GH login session**.

### 1 - Connect to the GH login node

```
ghlogin -A <project_id>
```

This places you inside the **Grace login environment**, which is the correct place to prepare and submit jobs targeting the `gh` partition.

Why this matters:

- The GH environment may have **different modules, paths, and architecture assumptions** (ARM/Grace).
- Submitting from the correct login environment ensures your job targets the **correct Slurm configuration and partition**.
- It avoids subtle issues where jobs are submitted from the wrong environment.

---

### 2 - Navigate to the directory containing the build script

For example:

```
cd /nobackup/projects/$PROJ/$USER/containers
```

This directory should contain:

- `build-vllm-sif.sbatch`
- `vllm-all-dep.def`

---

### 3 - Submit the build job

Run:

```
sbatch build-vllm-sif.sbatch
```

Slurm will then:

1. Place the job in the queue
2. Allocate a node in the `gh` partition
3. Execute the build script
4. Write logs to the file specified in the script.

---

### 4 - Monitor the job (optional)

You can check whether the job is queued or running with:

```
squeue -u $USER --start
```

Once finished, the log file will appear as:

```
build-vllm-sif-<jobid>.out
```

This file contains the full build output, including:

- Docker layer downloads
- Python package installation
- any errors encountered during the container build.

---

### Result

If the build completes successfully, the container image:

```
vllm-all-dep3.sif
```

will appear in the directory where the build command was executed (in this example, the `containers` folder in `/nobackup/projects/...`).

This `.sif` file is the **final Apptainer container** that can later be used to run vLLM jobs on Bede GH nodes.
