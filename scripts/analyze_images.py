"""Beginner-friendly one-command wrapper: normalize RGB PNGs, then run UNet++ inference."""
import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Normalize and segment your histology PNGs in one step")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "raw", help="Folder with raw RGB PNG images")
    parser.add_argument("--reference", type=Path, default=PROJECT_ROOT / "reference" / "normalization_reference.png", help="PNG reference used in training")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "models" / "best_foreground.pt", help="Pretrained model weights")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "inference", help="Folder for model results")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of patches in one inference batch (use 1 for low memory)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()
    raw, ref, ckpt = args.input, args.reference, args.checkpoint
    for path, label in [(raw, "Your input images folder"), (ref, "The normalization reference PNG"), (ckpt, "The trained model checkpoint")]:
        if not path.exists():
            parser.error(f"{label} is missing: {path}\nSee Steps 4-5 in README.md.")
    if not raw.is_dir():
        parser.error(f"Input path is not a folder: {raw}")
    if not list(raw.glob("*.png")):
        parser.error(f"No .png files found in {raw}. Copy your RGB PNG images there first.")
    if args.batch_size < 1:
        parser.error("--batch-size must be 1 or greater")
    norm = PROJECT_ROOT / "data" / "images_norm"
    scripts = PROJECT_ROOT / "scripts"
    subprocess.run([
        sys.executable, str(scripts / "normalize_images.py"), "--input", str(raw), "--reference", str(ref),
        "--output", str(norm)
    ], cwd=PROJECT_ROOT, check=True)
    subprocess.run([
        sys.executable, str(scripts / "run_inference.py"), "--images", str(norm),
        "--checkpoint", str(ckpt), "--output", str(args.output),
        "--batch-size", str(args.batch_size), "--device", args.device
    ], cwd=PROJECT_ROOT, check=True)
    print(f"\nFinished! Open: {args.output}")


if __name__ == "__main__":
    main()
