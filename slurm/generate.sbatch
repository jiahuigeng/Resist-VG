#!/bin/bash
#SBATCH --job-name=resist-vg
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# Add --account / --partition / -C (GPU type) here as your cluster requires, e.g.
# #SBATCH --account=<your-project>

# Usage (from the repo root):  sbatch slurm/generate.sbatch "your prompt here"
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Load conda; adjust for your cluster (module load Miniforge3, etc.)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate resist-vg

# Keep the ~20 GB model cache on project storage, not $HOME
export HF_HOME="${HF_HOME:-$PWD/models/hf}"

nvidia-smi
python generate.py --prompt "${1:-A red fox running through fresh snow, cinematic}"
