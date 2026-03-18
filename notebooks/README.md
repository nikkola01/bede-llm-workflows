# Notebooks — LLM Experiments

This folder contains a set of structured experiments designed to evaluate how large language models behave on Bede GPU nodes using vLLM.

Each notebook corresponds to a specific test, focusing on one aspect of model performance, behaviour, or system interaction.

---

## Test Overview

| Test | Main question | Parameters to vary | Fixed parameters | Metrics recorded | Notes |
| --- | --- | --- | --- | --- | --- |
| 1. Smoke test | Does the workflow run successfully? | model, prompt | temperature, max_tokens, GPU count | success/fail, startup time, inference time, output text | Keep this minimal and stable |
| 2. Prompt length reaction | How does prompt size affect runtime and behaviour? | prompt length, input token count | same model, same temperature, same max_tokens | total runtime, time to first token if available, memory use, output quality | Very useful for practical HPC guidance |
| 3. Output length test | How does requested output size affect performance? | max_tokens | same prompt, same model, same temperature | runtime, response completeness, truncation, memory use | Helps choose realistic generation limits |
| 4. Temperature test | How stable/variable are outputs? | temperature, repeat run number | same prompt, same model, same max_tokens | output variation, usefulness, factual stability | Good for showing deterministic vs creative use |
| 5. Model comparison | What is the tradeoff between model size and usefulness? | model name, tensor_parallel_size if needed | same prompt, same temperature, same max_tokens | load time, inference time, memory usage, answer quality | Important for selecting models on Bede |

---

## Notes

- All tests are designed to be:
  - reproducible  
  - comparable  
  - minimal in scope  

- Results should be interpreted together, not in isolation.

- These experiments are intended to support:
  - model selection  
  - parameter tuning  
  - efficient HPC usage  

> The goal is not just performance, but understanding how models behave in real HPC environments.
