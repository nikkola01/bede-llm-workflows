
# Running LLMs on Bede (GH Nodes)

This repository provides practical workflows for running large language models (LLMs) on the Bede HPC system, with a focus on GPU (GH) nodes.

It includes:
- container-based deployment using vLLM  
- native execution using llama.cpp  
- reproducible SLURM workflows  
- structured experiments for evaluating model behaviour  

---

## Structure
```
bede-llm-workflows/
│
├── docs/
│   ├── container_setup.md
│   ├── running_server.md
│   ├── running_python.md
│   ├── multi_gpu.md
│   ├── choosing_llm.md
│   ├── setup-lamma-ccp.md
│   └── README.md
│
├── containers/
│   ├── vllm-26.01-py3.def
│   ├── build_container.sbatch
│   └── README.md
│
├── slurm/
│   ├── run_vllm_inference.sbatch
│   ├── run_vllm_server.sbatch
│   ├── run_multi_gpu_inference.sbatch
│   └── README.md
│
├── notebooks/
│   ├── 01_smoke_test.ipynb
│   ├── 02_prompt_length_test.ipynb
│   ├── 03_output_length_test.ipynb
│   ├── 04_temperature_test.ipynb
│   ├── 05_model_comparison.ipynb
│   └── README.md
│
└── case_study/
    └── N8CIR_case_study.md
```
---

## What this repo enables

- Running LLM inference on HPC (batch jobs or server mode)  
- Comparing models, parameters, and performance  
- Integrating LLMs into research workflows  
- Reproducible AI experiments on GPU infrastructure  

---

## Recommended workflow

1. Read setup guide:
   - `docs/container_setup.md`

2. Choose a model:
   - `docs/choosing_llm.md`

3. Run a simple job:
   - `docs/running_python.md`

4. Run a server:
   - `docs/running_server.md`

5. Explore experiments:
   - `notebooks/`

---

## Approaches

### vLLM (container-based)
- high performance  
- supports large models  
- suitable for research pipelines and server workflows  

### llama.cpp (native)
- lightweight  
- easier to set up  
- suitable for smaller models and quick tests  

---

## Key considerations

- GPU compatibility (CUDA, drivers) is critical  
- containerisation avoids most dependency issues  
- model size must match available GPU memory  
- `/nobackup` should be used for all heavy workloads  

---

## Experiments

The `notebooks/` folder contains structured tests exploring:

- prompt length vs performance  
- output length scaling  
- temperature effects  
- model size trade-offs  

These are designed to support **practical decision-making**, not just benchmarking.

---

## Reproducibility

The workflows aim to be:
- consistent across runs  
- portable across users  
- aligned with HPC best practices  

---

## Final note

This repository is not a polished framework, but a **working, evolving set of tools and experiments**.

> The goal is to reduce friction when using LLMs on HPC, while staying honest about the limitations.
