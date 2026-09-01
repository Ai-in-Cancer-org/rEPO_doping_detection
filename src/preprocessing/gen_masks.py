"""
Generate binary foreground masks for extracted tiles.

A pixel is labelled foreground if it satisfies either a chromatic criterion in
HSV colour space or an edge criterion based on Sobel gradient magnitude. The two
maps are combined by logical union, refined by morphological closing and opening
with a 5x5 structuring element, and contour-filled to produce solid cell
regions. Masks are saved alongside their parent tiles for use as a fourth input
channel during training.

Usage:
    python src/preprocessing/gen_masks.py \
        --tile_root    data/tiles \
        --mask_root    data/masks \
        --overlay_root data/overlays
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse

def generate_foreground_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    raw_mask = ((sat > 25) & (val > 80)).astype(np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobelx**2 + sobely**2)
    edge_mask = (gradient > 40).astype(np.uint8)

    combined = np.clip(raw_mask + edge_mask, 0, 1)

    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    contours, hierarchy = cv2.findContours(cleaned, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(cleaned)

    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            if hierarchy[0][i][3] == -1:
                cv2.drawContours(filled, contours, i, 1, thickness=cv2.FILLED)
            else:
                cv2.drawContours(filled, contours, i, 0, thickness=cv2.FILLED)

    return (filled * 255).astype(np.uint8)


def create_overlay(image, mask):
    overlay = image.copy()
    red = np.zeros_like(image)
    red[:, :, 0] = 255
    alpha = 0.35

    binary_mask = mask.astype(bool)
    overlay[binary_mask] = cv2.addWeighted(
        image[binary_mask], 1 - alpha,
        red[binary_mask], alpha,
        0
    )
    return overlay


def process_single_slide(tile_dir, mask_root, overlay_root):
    tile_dir = os.path.abspath(tile_dir)
    slide_name = os.path.basename(tile_dir)

    mask_dir = os.path.join(mask_root, slide_name)
    overlay_dir = os.path.join(overlay_root, slide_name)

    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    tiles = sorted([f for f in os.listdir(tile_dir) if f.lower().endswith(".png")])

    print(f"\n📁 Slide: {slide_name}")
    print(f"   Tiles found: {len(tiles)}")
    print(f"   Saving masks → {mask_dir}")
    print(f"   Saving overlays → {overlay_dir}")

    for fname in tqdm(tiles, desc=f"{slide_name}: masks+overlays"):
        input_path = os.path.join(tile_dir, fname)
        mask_path = os.path.join(mask_dir, fname.replace(".png", "_mask.png"))
        overlay_path = os.path.join(overlay_dir, fname.replace(".png", "_overlay.png"))

        img_bgr = cv2.imread(input_path)

        if img_bgr is None:
            print(f"⚠️ Failed to read: {input_path}  (skipped)")
            continue

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        try:
            mask = generate_foreground_mask(img)
            cv2.imwrite(mask_path, mask)

            overlay = create_overlay(img, mask)
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")


def process_all(tile_root, mask_root, overlay_root):
    tile_root = os.path.abspath(tile_root)

    slide_dirs = sorted([
        os.path.join(tile_root, d)
        for d in os.listdir(tile_root)
        if os.path.isdir(os.path.join(tile_root, d))
    ])

    print(f"🔍 Found {len(slide_dirs)} slide folders inside {tile_root}")

    for slide_dir in slide_dirs:
        slide_name = os.path.basename(slide_dir)

        mask_dir = os.path.join(mask_root, slide_name)
        overlay_dir = os.path.join(overlay_root, slide_name)

        # 🔥 SKIP if both mask + overlay already exist AND are non-empty
        if os.path.isdir(mask_dir) and os.path.isdir(overlay_dir):
            if len(os.listdir(mask_dir)) > 0 and len(os.listdir(overlay_dir)) > 0:
                print(f"\n⏩ Skipping {slide_name} — outputs already exist")
                continue

        # Otherwise, generate
        process_single_slide(slide_dir, mask_root, overlay_root)

    print("\n✅ All done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch: Generate masks + overlays for ALL slide tile folders.")
    parser.add_argument("--tile_root", required=True, help="Folder containing many slide tile folders")
    parser.add_argument("--mask_root", required=True, help="Output root folder for masks")
    parser.add_argument("--overlay_root", required=True, help="Output root folder for overlays")
    parser.add_argument("--slide", default=None, help="Process only this slide (e.g. 22p4a)")

    args = parser.parse_args()

    if args.slide is not None:
            slide_dir = os.path.join(args.tile_root, args.slide)
            assert os.path.isdir(slide_dir), f"Slide not found: {slide_dir}"
            process_single_slide(slide_dir, args.mask_root, args.overlay_root)
    else:
        process_all(args.tile_root, args.mask_root, args.overlay_root)