#!/bin/bash
#SBATCH --job-name=build-vllm-sif
#SBATCH --partition=gh
#SBATCH --time=01:00:00
#SBATCH -A <project_id>
#SBATCH --mem=32G
#SBATCH -o build-vllm-sif-%j.out

set-euo pipefail

echo"[INFO] Building SIF on$(hostname) at$(date)"

export PROJ=<project_id>
export APPTAINER_CACHEDIR=/nobackup/projects/$PROJ/$USER/apptainer-cache
export APPTAINER_TMPDIR=/nobackup/projects/$PROJ/$USER/apptainer-tmp

mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

cd /nobackup/projects/$PROJ/$USER/containers

apptainer build vllm-26.01-py3.sif vllm-26.01-py3.def

echo "[INFO] Build finished at$(date)"
