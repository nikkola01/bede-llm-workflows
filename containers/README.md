# Containers (vLLM on Bede GH Nodes)

This folder contains Apptainer/Singularity container definitions and build scripts for running **vLLM on Bede GPU (GH) nodes**.

---

## Purpose

Running modern LLM frameworks (like vLLM) directly on HPC systems can be difficult due to:

- CUDA / driver compatibility issues  
- PyTorch / binary mismatches  
- complex dependency trees  

This container setup provides a **reproducible environment** that works reliably on Bede.

---

## Contents

| File | Description |
|------|------------|
| `*.def` | Container definition script |
| `build-*.sbatch` | SLURM scripts to build `.sif` images |
| `*.sif` | Built container images (not committed) |

---

## Important Note (vLLM + CUDA)

vLLM is **very sensitive to CUDA and PyTorch versions**.

We strongly recommend:

- Using the official NVIDIA base image

- Avoid installing vLLM manually on generic CUDA images  
  → this often leads to:
  - runtime crashes  
  - missing kernels  
  - incompatible binaries  

---

## Building a Container

Containers must be built on a compute node (not login node):

```bash
sbatch build-vllm-sif.sbatch
