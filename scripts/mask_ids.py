'''This script takes the annotated image from GIMP and sorts all the colors into three classes: background (0,0,0), soma (255,0,0), and neuropil (255, 255, 0)
All pixels that are not red or black are neuropill. Then, as a second pass, any cluster of red pixels smaller than
MIN_SOMA_SIZE is relabeled as neuropil (yellow), since real somas are never that small.
Finally it computes the class ids and saves the resulting mask as a png file ready for training.'''

import numpy as np
import cv2
import os
import glob
import argparse

MIN_SOMA_SIZE = 4  # red clusters with fewer pixels than this become yellow


def convert_colors_to_class(img):
    """
    Keeps red (255,0,0) and black (0,0,0) pixels.
    Changes all other pixels to yellow.
    Input:
        img : RGB image loaded with cv2.imread()
    Output:
        recolored image (RGB)
    """
    red_mask = np.all(img == [255, 0, 0], axis=-1)  # RGB red Syntax: for each (H, W) location, look at all channel values and reduce them to a single value (true or false) depending on whether all channel values match [255, 0, 0].
    black_mask = np.all(img == [0, 0, 0], axis=-1)  # RGB black
    other_mask = ~(red_mask | black_mask)  # whatever is not (~) red or black (|)
    img[other_mask] = [255, 255, 0]  # Change other pixels to yellow


def remove_small_red_regions(img, min_size=MIN_SOMA_SIZE, connectivity=8):
    """
    Second pass to catch labeling mistakes.
    Finds every connected cluster of red pixels and, if the cluster has
    fewer than `min_size` pixels, recolors it yellow (neuropil).
    Modifies img in place, like convert_colors_to_class.
    Input:
        img          : RGB image that has already gone through convert_colors_to_class()
        min_size     : smallest number of pixels a real soma can have
        connectivity : 8 = pixels touching diagonally count as the same cluster
                       4 = only pixels touching up/down/left/right count
    Output:
        number of red clusters that were changed to yellow
    """
    red_mask = np.all(img == [255, 0, 0], axis=-1).astype(np.uint8)  # 1 where red, 0 elsewhere

    # labels: HxW array where each red cluster gets its own number (0 = not red)
    # stats[i, cv2.CC_STAT_AREA] = number of pixels in cluster i
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(red_mask, connectivity=connectivity)

    areas = stats[:, cv2.CC_STAT_AREA]
    small_labels = np.where(areas < min_size)[0]
    small_labels = small_labels[small_labels != 0]  # label 0 is the non-red area, never touch it

    small_mask = np.isin(labels, small_labels)
    img[small_mask] = [255, 255, 0]  # tiny red clusters -> yellow

    return len(small_labels)


def label_rgb_to_ids(label_rgb: np.ndarray) -> np.ndarray:
    """
    Converts RGB label image with exact colors into class IDs.
    0 = background (black)
    1 = neuropil (yellow)
    2 = soma (red)
    """
    ids = np.zeros(label_rgb.shape[:2], dtype=np.uint8)  # initialize all to background (0). This image will be grayscale because we only take the first two dimensions (height, width)

    # neuropil (yellow)
    ids[np.all(label_rgb == [255, 255, 0], axis=-1)] = 1

    # soma (red)
    ids[np.all(label_rgb == [255, 0, 0], axis=-1)] = 2

    return ids


def main():
    parser = argparse.ArgumentParser(description="Convert RGB GIMP masks to IDs: 0=bg,1=neuropil,2=soma")
    parser.add_argument("--input", required=True, help="Directory of RGB annotation PNG files")
    parser.add_argument("--output", required=True, help="Destination for 0/1/2 grayscale PNG masks")
    parser.add_argument("--min-soma-size", type=int, default=MIN_SOMA_SIZE)
    args = parser.parse_args()
    if args.min_soma_size < 1:
        parser.error("--min-soma-size must be >= 1")
    os.makedirs(args.output, exist_ok=True)
    matches = sorted(glob.glob(os.path.join(args.input, '*.png')))
    if not matches:
        parser.error("No PNG files found in input directory")
    for path in matches:
        label_bgr = cv2.imread(path)
        if label_bgr is None:
            print(f"Skipped unreadable image: {path}")
            continue
        rgb = cv2.cvtColor(label_bgr, cv2.COLOR_BGR2RGB)
        convert_colors_to_class(rgb)
        removed = remove_small_red_regions(rgb, min_size=args.min_soma_size)
        mask = label_rgb_to_ids(rgb)
        target = os.path.join(args.output, os.path.basename(path))
        if not cv2.imwrite(target, mask):
            raise OSError(f"Cannot write {target}")
        print(f"{os.path.basename(path)} -> {target}; {removed} small soma groups relabeled")


if __name__ == "__main__":
    main()


'''Will uncomment this function if you want to use the squared distance method instead of direct matching.
But for now, direct matching is sufficient since we have only three distinct colors.'''
# # RGB colors & their corresponding class ids
# COLORS = np.array([
#     [0,   0,   0  ],  # bg 1
#     [255, 255, 0  ],  # neuropil (yellow) 2
#     [255, 0,   0  ],  # soma (red) 3
# ], dtype=np.int16)
# def label_rgb_to_ids(label_rgb: np.ndarray) -> np.ndarray:
#     """
#     label_rgb: HxWx3 uint8 in RGB
#     returns: HxW uint8 class ids {0,1,2}
#     """
#     lab = label_rgb.astype(np.int16)
#     # compute squared distance to each color
#     dists = np.sum((lab[..., None, :] - COLORS[None, None, :, :])**2, axis=-1)  # HxWx3 "Which of my known class colors is this pixel closest to?"
#     ids = np.argmin(dists, axis=-1).astype(np.uint8)
#     return ids