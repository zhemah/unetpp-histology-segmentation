"""
run_inference.py
----------------
Standalone full-slide inference script for UNet++ EfficientNet-B0.

Automatically runs inference on every .png found in PATH_NORMALIZED.

Outputs per image:
  <image_id>_pred_rgb.png   — color-coded segmentation mask
  <image_id>_overlay.png    — prediction blended over original image

Classes:
  0 = background  (black)
  1 = neuropil    (yellow)
  2 = soma        (red)

Usage:
  python scripts/run_inference.py --images data/images_norm --checkpoint models/best_foreground.pt

Pass image folder, checkpoint and output paths as command-line arguments.
"""

import argparse
import os
import math
import numpy as np
import cv2
import torch # https://pytorch.org/get-started/locally/
import segmentation_models_pytorch as smp  #pip install segmentation-models-pytorch


# These settings match the original training/inference workflow.
PATCH_SIZE = 512
STRIDE = 256
INFER_BATCH_SIZE = 4
NUM_CLASSES = 3
OVERLAY_ALPHA = 0.35

# =============================================================================
# Class colormap
# =============================================================================

BG_ID   = 0
NEU_ID  = 1
SOMA_ID = 2

ID2RGB = {
    BG_ID:   (0,   0,   0),
    NEU_ID:  (255, 255, 0),
    SOMA_ID: (255, 0,   0),
}

CLASS_NAMES = ["background", "neuropil", "soma"]


# =============================================================================
# Utilities (copied verbatim from training notebook)
# =============================================================================

def compute_pad_to_multiple(h, w, multiple=512):
    new_h = math.ceil(h / multiple) * multiple
    new_w = math.ceil(w / multiple) * multiple
    pad_h, pad_w = new_h - h, new_w - w
    top,  bottom = pad_h // 2, pad_h - pad_h // 2
    left, right  = pad_w // 2, pad_w - pad_w // 2
    return (top, bottom, left, right), (new_h, new_w)


def generate_patch_grid(h, w, patch_size=512, stride=256):
    xs = list(range(0, w - patch_size + 1, stride))
    ys = list(range(0, h - patch_size + 1, stride))
    return [(x, y) for y in ys for x in xs]


def pad_with_known_pad(img, pad, is_mask=False):
    top, bottom, left, right = pad
    if is_mask:
        return cv2.copyMakeBorder(img, top, bottom, left, right,
                                  borderType=cv2.BORDER_CONSTANT, value=0)
    return cv2.copyMakeBorder(img, top, bottom, left, right,
                              borderType=cv2.BORDER_CONSTANT, value=255)


def unpad_np(arr, pad):
    top, bottom, left, right = pad
    h, w = arr.shape[:2]
    return arr[top: h - bottom if bottom > 0 else h,
               left: w - right  if right  > 0 else w]


def ids_to_rgb(mask_ids):
    out = np.zeros((*mask_ids.shape, 3), dtype=np.uint8)
    for k, rgb in ID2RGB.items():
        out[mask_ids == k] = rgb
    return out


def overlay_pred_on_image(img_rgb, mask_ids, alpha=OVERLAY_ALPHA):
    mask_rgb = ids_to_rgb(mask_ids).astype(np.float32)
    out = (1 - alpha) * img_rgb.astype(np.float32) + alpha * mask_rgb
    return np.clip(out, 0, 255).astype(np.uint8)


# =============================================================================
# Model loading
# =============================================================================

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(num_classes=NUM_CLASSES):
    return smp.UnetPlusPlus(
        encoder_name    = "efficientnet-b0",
        encoder_weights = None,          # weights come from the checkpoint
        in_channels     = 3,
        classes         = num_classes,
    )


def load_checkpoint(model, ckpt_path, device):
    print(f"Loading checkpoint: {ckpt_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Download your trained weights first.")
    try:
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("Please use a modern PyTorch release supporting weights_only=True.") from exc

    if isinstance(obj, dict):
        if "model_state_dict" in obj:
            obj = obj["model_state_dict"]
        elif "state_dict" in obj:
            obj = obj["state_dict"]

    model.load_state_dict(obj, strict=True)
    model.to(device)
    model.eval()
    print(f"  Loaded on {device}")
    return model


# =============================================================================
# Full-slide inference
# =============================================================================

@torch.no_grad()
def predict_full_slide(model, img_rgb, pad, device,
                       patch_size=PATCH_SIZE, stride=STRIDE,
                       num_classes=NUM_CLASSES,
                       infer_batch_size=INFER_BATCH_SIZE):
    """
    Tile the full image, run the model, average overlapping logits, return
    per-pixel class predictions (original image dimensions, no padding).
    """
    img_pad      = pad_with_known_pad(img_rgb, pad, is_mask=False)
    h_pad, w_pad = img_pad.shape[:2]
    coords       = generate_patch_grid(h_pad, w_pad,
                                       patch_size=patch_size, stride=stride)

    logits_sum = np.zeros((num_classes, h_pad, w_pad), dtype=np.float32)
    count_sum  = np.zeros((h_pad, w_pad),              dtype=np.float32)

    total_batches = math.ceil(len(coords) / infer_batch_size)
    for batch_idx, i in enumerate(range(0, len(coords), infer_batch_size)):
        batch_coords = coords[i:i + infer_batch_size]
        patches = [img_pad[y:y + patch_size, x:x + patch_size]
                   for (x, y) in batch_coords]

        batch_t = (torch.from_numpy(np.stack(patches, axis=0))
                   .permute(0, 3, 1, 2).float() / 255.0).to(device)

        logits = model(batch_t).detach().cpu().numpy().astype(np.float32)

        for b, (x, y) in enumerate(batch_coords):
            logits_sum[:, y:y + patch_size, x:x + patch_size] += logits[b]
            count_sum[y:y + patch_size, x:x + patch_size]     += 1.0

        # Simple progress indicator
        if (batch_idx + 1) % max(1, total_batches // 10) == 0 or (batch_idx + 1) == total_batches:
            print(f"  Batch {batch_idx + 1}/{total_batches}", end="\r")

    print()  # newline after progress
    logits_avg = logits_sum / np.maximum(count_sum[np.newaxis, ...], 1e-6)
    pred_pad   = np.argmax(logits_avg, axis=0).astype(np.uint8)
    return unpad_np(pred_pad, pad)


def run_inference_on_image(image_id, model, device, path_normalized, out_dir,
                           patch_size=PATCH_SIZE, stride=STRIDE,
                           infer_batch_size=INFER_BATCH_SIZE):
    """
    Load one image, run full-slide inference, and save:
      - <image_id>_pred_rgb.png  (color-coded segmentation)
      - <image_id>_overlay.png   (prediction blended over image)
    """
    img_path = os.path.join(path_normalized, f"{image_id}.png")
    img_bgr  = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    h, w = img_rgb.shape[:2]
    pad, _ = compute_pad_to_multiple(h, w, multiple=patch_size)

    print(f"\n[{image_id}]  size=({h}, {w})  pad={pad}")
    pred_ids = predict_full_slide(model, img_rgb, pad, device,
                                  patch_size=patch_size, stride=stride,
                                  infer_batch_size=infer_batch_size)

    os.makedirs(out_dir, exist_ok=True)

    pred_rgb = ids_to_rgb(pred_ids)
    overlay  = overlay_pred_on_image(img_rgb, pred_ids)

    pred_ids_path = os.path.join(out_dir, f"{image_id}_pred_ids.png")
    pred_rgb_path = os.path.join(out_dir, f"{image_id}_pred_rgb.png")
    overlay_path  = os.path.join(out_dir, f"{image_id}_overlay.png")

    cv2.imwrite(pred_ids_path, pred_ids)
    cv2.imwrite(pred_rgb_path, cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(overlay_path,  cv2.cvtColor(overlay,  cv2.COLOR_RGB2BGR))

    print(f"  Saved: {pred_ids_path}")
    print(f"  Saved: {pred_rgb_path}")
    print(f"  Saved: {overlay_path}")

    unique, counts = np.unique(pred_ids, return_counts=True)
    total_px = pred_ids.size
    for cls_id, cnt in zip(unique, counts):
        print(f"  Class {cls_id} ({CLASS_NAMES[cls_id]}): "
              f"{cnt:,} px  ({100 * cnt / total_px:.1f}%)")


# =============================================================================
# Main
# =============================================================================

def scan_images(folder):
    """Return sorted list of image IDs (stems) for every .png in folder."""
    ids = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(folder)
        if f.lower().endswith(".png")
    )
    if not ids:
        raise FileNotFoundError(f"No .png files found in: {folder}")
    return ids


def main():
    parser = argparse.ArgumentParser(description="UNet++ RGB histology segmentation (3 classes).")
    parser.add_argument("--images", required=True, help="Folder of normalized RGB PNG images")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt state-dict checkpoint")
    parser.add_argument("--output", default="outputs/inference", help="Where to save outputs")
    parser.add_argument("--batch-size", type=int, default=INFER_BATCH_SIZE)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()
    if not os.path.isdir(args.images):
        parser.error(f"Image directory does not exist: {args.images}")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    device = get_device() if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device}")
    model = load_checkpoint(build_model(), args.checkpoint, device)
    ids = scan_images(args.images)
    print(f"Found {len(ids)} PNG image(s)")
    for image_id in ids:
        run_inference_on_image(image_id, model, device, args.images, args.output,
                               infer_batch_size=args.batch_size)
    print(f"\nAll done. Outputs saved to: {args.output}")


if __name__ == "__main__":
    main()
