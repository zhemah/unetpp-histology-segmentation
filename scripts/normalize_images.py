'''Normalizes brain-section immunoperoxidase (DAB) images to a reference image before UNet++ training / inference.

Why not histogram matching?
  Histogram matching forces the target's intensity distribution to equal the reference's. If a section has
  much more real staining than the reference (e.g. a big GFP+ tract), the matching treats that staining as a
  brightness error and pulls it up to neuropil colors -> the signal gets erased.

How this version works (robust to HOW MUCH of the tissue is stained):
  1. Find the tissue (everything darker than the white slide background) and the slide background color
     (= white point, corrects illumination / white balance).
  2. Build an "interior" mask (tissue >= EDGE_MARGIN px from section edges and large holes), so the dark rim
     does not influence the fit.
  3. Convert to optical density (OD = -log10(I / white)), where stain amount is ~linear.
  4. Anchor = OD of the lightest NEUROPIL_PERCENTILE % of interior tissue (unstained neuropil), per channel.
     This is almost always unstained tissue, no matter how much stain the section has.
  5. Scale each channel's OD so the target's neuropil anchor equals the reference's, then convert back to RGB
     using the reference white point. It is one multiplication in OD space, so the order and relative
     darkness of everything is preserved: dark stays dark, heavily stained sections stay heavily stained.
  6. Applied to the whole image via a 256-entry lookup table per channel (fast, low memory). Background is
     mapped to the reference's background color, so there is no seam at the tissue edge.

Use the exact same settings for training images and for inference images.
'''
import cv2 as cv
import numpy as np
import os
import argparse
from tqdm import tqdm
# ---------------------------------------------------------------- settings
# The reference image must be supplied with --reference at runtime.
EDGE_MARGIN = 100          # px. Tissue closer than this to an edge/large hole is not used to FIT the normalization
MIN_HOLE_AREA = 20500      # px^2. Background regions inside the tissue bigger than this count as real holes/tears
MIN_TISSUE_AREA = 5000     # px^2. Tissue pieces smaller than this (debris, specks) are ignored for fitting
NEUROPIL_PERCENTILE = 25   # % lightest interior tissue used as the "unstained neuropil" anchor.
                           # Must be smaller than the fraction of interior tissue that is unstained (25 is safe for
                           # almost every section; lower it to 10-15 if some sections are stained nearly everywhere)
MAX_SCALE_CHANGE = 2.0     # safety clamp: OD is never scaled by more than x2 or less than x0.5
MATCH_STRENGTH = 1.0       # 1.0 = full normalization, 0.5 = halfway between original and normalized, 0 = off
SAVE_QC = False            # True -> also saves a picture of the fitting region for each image (to check EDGE_MARGIN)


def find_background_threshold(hist):
    """
    FOR WHITE BACKGROUNDS ONLY
    Finds the background threshold by detecting the largest dip after the histogram peak.
    """
    peak_intensity = np.argmax(hist[200:]) + 200
    threshold = peak_intensity
    min_value = float('inf')

    for i in range(peak_intensity, 150, -1):
        if hist[i] < hist[i - 1]:
            if hist[i] < min_value:
                min_value = hist[i]
                threshold = i

    return threshold


def get_tissue_mask(img_rgb):
    """
    Tissue = pixels darker than the white background threshold.
    Returns a boolean mask (True = tissue).
    """
    gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
    hist = cv.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    threshold = find_background_threshold(hist)
    return gray < threshold


def get_interior_mask(tissue_mask, edge_margin=EDGE_MARGIN, min_hole_area=MIN_HOLE_AREA, min_tissue_area=MIN_TISSUE_AREA):
    """
    Returns the part of the tissue that is at least `edge_margin` px away from:
      - the outer edge of the section
      - large holes/tears (area >= min_hole_area)
    Small bright spots inside the tissue (vessels, tiny holes) are NOT treated as edges.
    """
    tissue = tissue_mask.astype(np.uint8)

    # 1) Drop small tissue pieces (debris)
    n, labels, stats, _ = cv.connectedComponentsWithStats(tissue, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv.CC_STAT_AREA] >= min_tissue_area
    tissue = keep[labels].astype(np.uint8)

    # 2) Fill small background regions inside the tissue
    background = (1 - tissue).astype(np.uint8)
    n, labels, stats, _ = cv.connectedComponentsWithStats(background, connectivity=4)
    h, w = tissue.shape
    x, y, bw, bh, area = (stats[:, i] for i in range(5))
    touches_border = (x == 0) | (y == 0) | (x + bw == w) | (y + bh == h)
    small_hole = (area < min_hole_area) & ~touches_border
    small_hole[0] = False
    filled = tissue.copy()
    filled[small_hole[labels]] = 1

    # 3) Distance to nearest real edge / large hole (image frame counts as an edge)
    padded = cv.copyMakeBorder(filled, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0)
    dist = cv.distanceTransform(padded, cv.DIST_L2, 5)[1:-1, 1:-1]
    interior = (dist > edge_margin) & tissue_mask

    return interior


def get_fit_mask(tissue_mask, min_fraction=0.10):
    """
    Interior mask with a fallback: if the interior is less than `min_fraction` of the tissue,
    the margin is halved until there are enough pixels (last resort: all tissue).
    """
    n_tissue = int(tissue_mask.sum())
    margin = EDGE_MARGIN
    while margin >= 10:
        interior = get_interior_mask(tissue_mask, edge_margin=margin)
        if interior.sum() >= min_fraction * n_tissue:
            return interior, margin
        margin //= 2
    return tissue_mask.copy(), 0


# ---------------------------------------------------------------- OD-anchor normalization
def get_white_point(img_rgb, tissue_mask):
    """Slide background color per channel (median of non-tissue pixels). Fallback if there is almost no background."""
    background = ~tissue_mask
    if background.sum() >= 1000:
        return np.median(img_rgb[background], axis=0).astype(np.float64)
    return np.percentile(img_rgb.reshape(-1, 3), 99.5, axis=0).astype(np.float64)


def intensity_to_od(values, white):
    """Optical density relative to the white point. Pixels brighter than the white point get OD 0."""
    values = np.asarray(values, dtype=np.float64)
    return -np.log10(np.clip((values + 1.0) / (white + 1.0), 1e-6, 1.0))


def get_neuropil_anchor(img_rgb, fit_mask, white, percentile=NEUROPIL_PERCENTILE):
    """
    OD of unstained neuropil, per channel = OD at the lightest `percentile` % of interior tissue.
    (Lightest p% in OD = the (100 - p)th percentile of intensity.)
    """
    anchor = np.empty(3)
    for c in range(3):
        light_value = np.percentile(img_rgb[..., c][fit_mask], 100 - percentile)
        anchor[c] = intensity_to_od(light_value, white[c])
    return np.maximum(anchor, 0.005)  # avoid dividing by ~0 if neuropil is almost white


def build_luts(target_white, target_anchor, reference_white, reference_anchor, strength=MATCH_STRENGTH):
    """One 256-entry LUT per channel: intensity -> OD -> scaled OD -> intensity under the reference white point."""
    scale = np.clip(reference_anchor / target_anchor, 1 / MAX_SCALE_CHANGE, MAX_SCALE_CHANGE)
    all_vals = np.arange(256, dtype=np.float64)
    luts = []
    for c in range(3):
        od = intensity_to_od(all_vals, target_white[c]) * scale[c]
        mapped = (reference_white[c] + 1.0) * 10.0 ** (-od) - 1.0
        lut = all_vals + strength * (mapped - all_vals)
        luts.append(np.clip(np.round(lut), 0, 255).astype(np.uint8))
    return luts, scale


def apply_luts(img_rgb, luts):
    out = np.empty_like(img_rgb)
    for c in range(3):
        out[..., c] = cv.LUT(img_rgb[..., c], luts[c])
    return out


def save_qc(img_rgb, tissue_mask, fit_mask, path):
    """Tissue used for fitting in green, tissue excluded from fitting (rim/edges) in magenta."""
    overlay = img_rgb.copy()
    overlay[fit_mask] = (0.6 * overlay[fit_mask] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
    excluded = tissue_mask & ~fit_mask
    overlay[excluded] = (0.6 * overlay[excluded] + 0.4 * np.array([255, 0, 255])).astype(np.uint8)
    cv.imwrite(path, cv.cvtColor(overlay, cv.COLOR_RGB2BGR))


def process_image(image_path, reference_white, reference_anchor, output_folder):
    target_img_bgr = cv.imread(image_path, cv.IMREAD_COLOR)
    if target_img_bgr is None:
        print(f"Error loading image: {image_path}")
        return
    target_img = cv.cvtColor(target_img_bgr, cv.COLOR_BGR2RGB)
    name = os.path.basename(image_path)

    target_tissue = get_tissue_mask(target_img)
    if target_tissue.sum() == 0:
        print(f"No tissue found, skipped: {image_path}")
        return
    target_fit, margin_used = get_fit_mask(target_tissue)
    if margin_used < EDGE_MARGIN:
        tqdm.write(f"  Note: {name} is small/narrow, fitted with margin {margin_used}px")

    target_white = get_white_point(target_img, target_tissue)
    target_anchor = get_neuropil_anchor(target_img, target_fit, target_white)
    luts, scale = build_luts(target_white, target_anchor, reference_white, reference_anchor)
    normalized = apply_luts(target_img, luts)

    # Scale factors near 1 = image already looks like the reference. Values hitting the clamp deserve a look.
    hit_clamp = np.isclose(scale, MAX_SCALE_CHANGE) | np.isclose(scale, 1 / MAX_SCALE_CHANGE)
    flag = "  <-- hit clamp, check this image" if hit_clamp.any() else ""
    tqdm.write(f"  {name}: OD scale R/G/B = {scale[0]:.2f} / {scale[1]:.2f} / {scale[2]:.2f}{flag}")

    original_file_name = os.path.splitext(name)[0]
    output_path = os.path.join(output_folder, f"{original_file_name}.png")
    cv.imwrite(output_path, cv.cvtColor(normalized, cv.COLOR_RGB2BGR))

    if SAVE_QC:
        qc_folder = os.path.join(output_folder, 'QC fit region')
        os.makedirs(qc_folder, exist_ok=True)
        save_qc(target_img, target_tissue, target_fit, os.path.join(qc_folder, f"{original_file_name}_fit.png"))


def main():
    parser = argparse.ArgumentParser(description="Normalize tissue images against a reference RGB PNG.")
    parser.add_argument("--input", required=True, help="Directory with source PNG images")
    parser.add_argument("--reference", required=True, help="Reference image PNG used in original training")
    parser.add_argument("--output", default=None, help="Output folder (default: INPUT/normalized)")
    parser.add_argument("--save-qc", action="store_true", help="Save tissue-fitting QC overlays")
    args = parser.parse_args()
    if not os.path.isdir(args.input):
        parser.error(f"Input directory not found: {args.input}")
    if not os.path.isfile(args.reference):
        parser.error(f"Reference PNG not found: {args.reference}")
    output_folder = args.output or os.path.join(args.input, "normalized")
    if os.path.abspath(args.input) == os.path.abspath(output_folder):
        parser.error("Output must be a different directory from input")
    global SAVE_QC
    SAVE_QC = args.save_qc
    os.makedirs(output_folder, exist_ok=True)
    reference_img_bgr = cv.imread(args.reference, cv.IMREAD_COLOR)
    if reference_img_bgr is None:
        raise ValueError("Could not read reference image")
    reference_img = cv.cvtColor(reference_img_bgr, cv.COLOR_BGR2RGB)
    reference_tissue = get_tissue_mask(reference_img)
    if reference_tissue.sum() == 0:
        raise ValueError("No tissue detected in reference image")
    reference_fit, ref_margin = get_fit_mask(reference_tissue)
    reference_white = get_white_point(reference_img, reference_tissue)
    reference_anchor = get_neuropil_anchor(reference_img, reference_fit, reference_white)
    print(f"Reference: {int(reference_fit.sum())} interior tissue pixels "
          f"(margin {ref_margin}px); white={reference_white.round()}, OD={reference_anchor.round(3)}")
    filenames = sorted(f for f in os.listdir(args.input) if f.lower().endswith('.png')
                       and os.path.isfile(os.path.join(args.input, f)))
    if not filenames:
        raise ValueError(f"No PNG files in: {args.input}")
    for name in tqdm(filenames, desc="Normalizing"):
        process_image(os.path.join(args.input, name), reference_white, reference_anchor, output_folder)


if __name__ == "__main__":
    main()
