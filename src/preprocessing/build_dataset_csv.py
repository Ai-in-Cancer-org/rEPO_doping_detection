#!/usr/bin/env python3
"""
Build a dataset CSV from a tile directory and a slide-level label file.

Iterates over per-slide tile folders and assigns each tile the binary label of
its parent slide.

Usage:
    python src/preprocessing/build_dataset_csv.py \
        --tiles_dir data/tiles \
        --label_csv configs/slide_labels.csv \
        --out_csv   data/metadata/dataset.csv
"""


import argparse
import csv
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles_dir", required=True,
                    help="Root dir containing per-slide tile subfolders")
    ap.add_argument("--label_csv", required=True,
                    help="CSV with columns: slide_id, label")
    ap.add_argument("--out_csv",   required=True,
                    help="Output CSV path")
    ap.add_argument("--img_ext",   default=".png",
                    help="Tile image extension (default: .png)")
    args = ap.parse_args()

    tiles_dir = Path(args.tiles_dir)
    out_csv   = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Load slide labels
    labels_df = pd.read_csv(args.label_csv)
    assert "slide_id" in labels_df.columns and "label" in labels_df.columns, \
        "label_csv must have 'slide_id' and 'label' columns"
    label_map = dict(zip(labels_df["slide_id"].astype(str),
                         labels_df["label"].astype(int)))

    print(f"Label file: {len(label_map)} slides defined")
    print(f"Scanning tiles: {tiles_dir}")

    rows = []
    missing_label = []

    for slide_dir in sorted(tiles_dir.iterdir()):
        if not slide_dir.is_dir():
            continue
        slide_id = slide_dir.name
        if slide_id not in label_map:
            missing_label.append(slide_id)
            continue
        label = label_map[slide_id]
        for tile in sorted(slide_dir.glob(f"*{args.img_ext}")):
            rel = f"{slide_id}/{tile.name}"
            rows.append({
                "relative_img_path": rel,
                "label":   label,
                "slide_id": slide_id,
            })

    if missing_label:
        print(f"WARNING: {len(missing_label)} slide folders have no label "
              f"entry and were skipped: {missing_label[:5]}{'...' if len(missing_label)>5 else ''}")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["relative_img_path",
                                               "label", "slide_id"])
        writer.writeheader()
        writer.writerows(rows)

    n1 = sum(1 for r in rows if r["label"] == 1)
    n0 = sum(1 for r in rows if r["label"] == 0)
    slides = len(set(r["slide_id"] for r in rows))
    print(f"\nDone: {len(rows):,} tiles | {slides} slides | "
          f"positive={n1:,} negative={n0:,}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
