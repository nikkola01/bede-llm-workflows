# Documentation — Running LLMs on Bede (GH Nodes)

This folder contains step-by-step guides for deploying and running large language models (LLMs) on the Bede HPC system.

The documentation is structured to support both:
- **container-based workflows (vLLM)** → recommended for most users  
- **native workflows (llama.cpp)** → lightweight alternative without containers  

---

##  Contents

### Setup and Environment

| File | Description |
|------|------------|
| `container_setup.md` | How to build and configure Apptainer containers for vLLM |
| `setup-lamma-ccp.md` | Manual setup of llama.cpp without containers |

---

### Running Workloads

| File | Description |
|------|------------|
| `running_python.md` | Running inference using Python scripts (batch jobs) |
| `running_server.md` | Running a persistent vLLM server (API-based workflows) |

---

### Advanced Usage

| File | Description |
|------|------------|
| `multi_gpu.md` | Running across multiple GPUs (tensor parallelism) |
| `choosing_llm.md` | Guidance on selecting appropriate HuggingFace models |

---

## Workflow Overview

There are two main ways to run LLMs in this project:

### 1. Container-based (vLLM) — Recommended

- Uses NVIDIA vLLM containers  
- Supports:
  - large models  
  - server-based inference  
  - agent workflows  

✔ Best for:
- research workflows  
- scalable inference  
- reproducibility  

---

### 2. Native (llama.cpp)

- Runs directly on the system without containers  
- Lightweight and simple setup  

✔ Best for:
- small models  
- quick experiments  
- environments where containers are not suitable  

⚠ Limitations:
- limited GPU utilisation compared to vLLM  
- less suitable for large-scale inference  

---

## Suggested Learning Path

If you are new to the system, follow this order:

1. `container_setup.md`  
2. `choosing_llm.md`  
3. `running_python.md`  
4. `running_server.md`  
5. `multi_gpu.md`  

Optional:
- `setup-lamma-ccp.md` (alternative workflow)


### Containers vs Native

- vLLM (container):
  - more complex setup  
  - much higher performance and flexibility  

- llama.cpp:
  - easier setup  
  - lower performance ceiling  

---

## Final Note

This documentation is designed to reduce friction, but HPC + LLM systems remain complex.

If something does not work:
- check logs first  
- verify paths and permissions  
- confirm model compatibility  

And most importantly:

> Start simple, verify it works, then scale.
