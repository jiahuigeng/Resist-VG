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

## On a Slurm cluster

```bash
git clone https://github.com/<user>/Resist-VG.git && cd Resist-VG
# create the conda env as above, then either:

# a) interactive session
salloc --gpus=1 --time=01:00:00        # add --account/-C as your cluster needs
export HF_HOME=$PWD/models/hf
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
