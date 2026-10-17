# Resist-VG

Video generation with [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) via diffusers.
Default output: 1280×704, 121 frames (~5 s at 24 fps). Needs a GPU with ≥ 24 GB VRAM (or `--offload`).

## Setup

```bash
conda create -n resist-vg python=3.12 -y
conda activate resist-vg
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

The model (~20 GB) downloads from Hugging Face on first run into `$HF_HOME` (default `~/.cache/huggingface`).

## Run

```bash
python generate.py --prompt "A red fox running through fresh snow, cinematic"
python generate.py --steps 20 --frames 49     # quick test
```

Videos are written to `outputs/`.

## On Arrhenius (GH200, aarch64)

GPU nodes are ARM, so the env must be built on a GPU node. `scripts/arrhenius_setup.sh` installs
Miniforge + the `resist-vg` env under the project directory and runs a smoke test:

```bash
git clone https://github.com/jiahuigeng/Resist-VG.git && cd Resist-VG
srun -A naiss2026-3-658-gpu -p gpu --gpus 1 -c 16 -t 02:00:00 --pty bash scripts/arrhenius_setup.sh

# a) interactive session
interactive -A naiss2026-3-658-gpu --partition gpu --gpus 1 -t 02:00:00
source /nobackup/proj/disk/naiss2026-3-658/personal/jiage91/miniforge3/etc/profile.d/conda.sh
conda activate /nobackup/proj/disk/naiss2026-3-658/personal/jiage91/envs/resist-vg
python generate.py --prompt "..."

# b) batch job (keeps running after you log out)
mkdir -p logs
sbatch slurm/generate.sbatch "your prompt"
squeue --me
tail -f logs/resist-vg-<jobid>.out
```

For a long interactive session that survives SSH disconnects, start `tmux` on the login node first,
then `salloc` inside it; reattach later with `tmux attach`.

## Sync workflow

```bash
# local
git add -A && git commit -m "..." && git push
# cluster
git pull
```
