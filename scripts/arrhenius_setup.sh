#!/bin/bash
# Set up and smoke-test Resist-VG on an Arrhenius GH200 (aarch64) GPU node.
# Run on a GPU node from the repo root, e.g.:
#   srun -A naiss2026-3-658 -p gpu --gpus 1 -t 02:00:00 --pty bash scripts/arrhenius_setup.sh
# Idempotent: reuses Miniforge and the env if they already exist.
set -euo pipefail

BASE="${BASE:-/nobackup/proj/disk/naiss2026-3-658/personal/jiage91}"
CONDA_DIR="$BASE/miniforge3"
ENV_DIR="$BASE/envs/resist-vg"

echo "== node: $(hostname)  arch: $(uname -m)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  echo "== installing Miniforge ($(uname -m)) to $CONDA_DIR"
  tmp=$(mktemp -d)
  curl -fsSL -o "$tmp/miniforge.sh" \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh"
  bash "$tmp/miniforge.sh" -b -p "$CONDA_DIR"
  rm -rf "$tmp"
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"

if [ ! -d "$ENV_DIR" ]; then
  echo "== creating env $ENV_DIR"
  conda create -y -p "$ENV_DIR" -c conda-forge --override-channels python=3.12
fi
conda activate "$ENV_DIR"

# --no-cache-dir: keep the project's file-count quota free of pip caches
if ! python -c "import torch" 2>/dev/null; then
  pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128
fi
pip install --no-cache-dir -q -r requirements.txt

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "== smoke test"
python generate.py --prompt "A red fox running through fresh snow in a pine forest, cinematic lighting" \
  --steps 20 --frames 49
ls -lh outputs/
