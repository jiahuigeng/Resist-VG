"""Text-to-video generation with Wan2.2-TI2V-5B (diffusers).

Example:
    python generate.py --prompt "A cat surfing at sunset" --steps 50
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

# Official recommended negative prompt for Wan models.
DEFAULT_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt", default="Two anthropomorphic cats in comfy boxing gear and bright gloves "
                   "fight intensely on a spotlighted stage.")
    p.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE)
    p.add_argument("--model", default=MODEL_ID, help="HF repo id or local directory")
    p.add_argument("--height", type=int, default=704)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--frames", type=int, default=121, help="must be 4k+1; 121 frames = ~5 s at 24 fps")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--offload", action="store_true",
                   help="CPU offload to save VRAM (slower; use on GPUs with < 24 GB)")
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not available")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    t0 = time.time()
    vae = AutoencoderKLWan.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(args.model, vae=vae, torch_dtype=torch.bfloat16)
    # Decoding 720p in one pass needs >32 GB VRAM; tiling keeps it small.
    pipe.vae.enable_tiling()
    if args.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    print(f"Model loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    frames = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        generator=torch.Generator("cuda").manual_seed(args.seed),
    ).frames[0]
    print(f"Generated in {time.time() - t0:.1f}s, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{datetime.now():%Y%m%d-%H%M%S}_seed{args.seed}.mp4"
    export_to_video(frames, str(out), fps=args.fps)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
