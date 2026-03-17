# How to pick a Hugging Face LLM for Bede gh (Grace Hopper GH200) nodes

## Know what a “gh node” actually is on Bede

On Bede, `gh` nodes are **NVIDIA Grace Hopper superchips** (ARM CPU + Hopper GPU). The hardware page shows the key constraints: most `gh` nodes have **1× H100 96GB**, **72 Arm Neoverse V2 CPU cores**, and **480GB RAM**, plus there is also a special `gh` node with **2× GH200 (H100 144GB each) + 960GB RAM**.

Operationally:

- `ghlogin` gives you a **Grace CPU-only environment** (no GPU) for building/testing basics; for GPU testing you submit to `ghtest`.
- On `gh`, requesting `-gres=gpu:1` gives you the *whole* node (72 cores + 480GB RAM).
- Bede’s “Using Bede” page is here (same link you shared): https-bede-documentation.readth…

## 1) First decision: training vs inference

### If you want **training / fine-tuning**

- On a single GH200 (96GB), you’re usually in **LoRA/QLoRA** territory for >7B–13B models (depending on sequence length + batch size).
- Full fine-tuning [bede-documentation](https://bede-documentation.readthedocs.io/en/latest/) to become **multi-node** and then you’re fighting queue times + distributed setup complexity.

### If you want **inference**

You can comfortably run:

- **7B–34B** class models in **FP16/BF16** (depending on architecture + context length).
- **70B** class models typically require **quantization (4–8 bit)** or tensor-parallel across multiple GPUs (which is more involved on Bede `gh` because it’s usually 1 GPU per node).

## 2) Rule-of-thumb sizing for GH200 (96GB)

When picking a model on Hugging Face, start with parameter count and precision:

**Weights VRAM (very rough):**

- FP16/BF16: `~2 bytes × params`
    - 7B → ~14GB
    - 13B → ~26GB
    - 34B → ~68GB
    - 40B → ~80GB (often still feasible, but KV cache may push you over)
    - 70B → ~140GB (not feasible on 96GB without quantization)

Then add:

- **KV cache** (grows with context length and batch). Long context (e.g., 32k–128k) can dominate memory even if weights fit.
- Framework overhead (tens of %).

**Practical takeaway:** on 96GB, pick **≤34B** for comfortable BF16/FP16 inference at moderate context (4k–16k). For **70B**, plan **4-bit quant** (or use the 2×144GB node if you can land it). Hardware reference:

## 3) Second decision: your inference stack determines what “works” on ARM

Because `gh` is **aarch64**, the *weights* are portable, but your **runtime must support ARM + CUDA**.

Common options:

- **vLLM** (great throughput/serving, paged KV cache; check your vLLM build supports aarch64 + your CUDA).
- **Hugging Face Transformers** (simpler, great for experiments; slower than vLLM at scale).
- **TensorRT-LLM** (fast, but heavier setup).

On Bede you’ll often do this via **Apptainer** containers (common on HPC). Bede’s docs explicitly discuss compiling CUDA stubs on `ghlogin` and using `ghtest` for GPU testing.

(If you go container-based, make sure the image supports **aarch64** and **CUDA 12.x**.)

## 4) How to evaluate a Hugging Face model page (a checklist)

When you’re browsing candidate models on Hugging Face, check:

### A) License & access

- Is it Apache/MIT (easy), or “community” / “research only” / gated?
- If gated, make sure your HF token works non-interactively on the cluster.

### B) Architecture & features

- Decoder-only for chat (LLaMA/Mistral/Qwen-style) vs encoder-decoder (T5-style).
- Context length: 8k vs 32k/128k. Remember: higher context → higher KV cache → more VRAM pressure.

### C) Weight format & loadability

- Prefer **`safetensors`**.
- Check if the repo provides:
    - BF16/FP16 weights
    - Quantized variants (AWQ / GPTQ / GGUF) *if you plan quant inference*

### D) “Fits the GH200 sweet spot”

- On 96GB: aim **7B–34B** BF16/FP16 unless you know you’ll quantize.
- On the 2×144GB node: larger becomes possible, but it’s a single special node and may be harder to get.

### E) Community + maturity signals

- Recent updates, known working examples, issues not full of “doesn’t load” reports.
- If you’re using vLLM, check if the architecture is well-supported there.

## 5) A simple picking strategy (fast path)

If you just want a sane default workflow:

1. Pick **one “small” model** for pipeline validation (7B-ish).
2. Pick **one “mid” model** you expect to use for real runs (13B–34B).
3. Only then attempt **70B** with quantization if you still need it.

This avoids burning queue time debugging a giant model.

## 6) Slurm skeletons you’ll actually use on Bede GH

### Quick GPU test (use `ghtest`)

Bede’s docs show `ghtest` for short GPU tests and an example requesting `--gres=gpu:1`.

```
#!/bin/bash
#SBATCH --account=<project>
#SBATCH --partition=ghtest
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0:15:00

nvidia-smi
python-c"import torch; print(torch.cuda.get_device_name(0))"
```

### Full run (use `gh`)

```
#!/bin/bash
#SBATCH --account=<project>
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00

nvidia-smi
# load your env / apptainer exec ... / run vllm or transformers
```

Bede notes you submit to `gh`/`ghtest` either from inside `ghlogin`, or via helper commands (`ghbatch`/`ghrun`).

## 7) Common pitfalls (so you don’t lose a day)

- **Forgetting ARM:** your local x86 wheels/containers may not work on `gh`. Prefer aarch64-ready conda/env/containers.
- **Context length surprise:** a “fits at 8k” model can OOM at 32k even though weights fit.
- **Quantization mismatch:** GPTQ/AWQ/GGUF require specific runtimes; don’t assume “quantized” automatically works with Transformers.
- **Gated models on HPC:** test token auth early on `ghtest`.
