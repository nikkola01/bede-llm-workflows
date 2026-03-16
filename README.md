# Running Large Language Models on Bede HPC

This repository provides reproducible workflows for running
Large Language Models (LLMs) on the Bede HPC system using:

- Apptainer containers
- vLLM inference engine
- Slurm job scheduling

The project demonstrates how researchers can deploy and run
LLM workloads on GPU clusters with minimal setup.

## Features

- Containerised AI environment
- Example Slurm job scripts
- Python inference workflows
- Multi-GPU configuration
- Test notebooks demonstrating model behaviour

## Quick Start

1. Build the container
2. Submit a Slurm job
3. Run a Python inference script

See the `docs/` folder for full instructions.

## Documentation

Container setup: docs/container_setup.md  
Running inference: docs/running_inference.md  
Running server: docs/running_server.md

## Example Experiments

The `notebooks/` directory contains example tests evaluating:

- prompt length sensitivity
- generation parameters
- model comparisons

## Case Study

See `case_study/N8CIR_case_study.md`
