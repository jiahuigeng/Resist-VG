#!/bin/bash
#SBATCH --job-name=resist-vg
#SBATCH --account=naiss2026-3-658-gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

# Usage (from the repo root, after scripts/arrhenius_setup.sh has built the env):
#   mkdir -p logs && sbatch slurm/generate.sbatch "your prompt here"
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

BASE="${BASE:-/nobackup/proj/disk/naiss2026-3-658/personal/jiage91}"
source "$BASE/miniforge3/etc/profile.d/conda.sh"
conda activate "$BASE/envs/resist-vg"

nvidia-smi
python generate.py --prompt "${1:-A red fox running through fresh snow, cinematic}"
