#!/usr/bin/env python3
"""
Extract image tiles from whole slide images using quality-control output.

Reads per-slide quality-control score files produced by the monolayer detection
step and, for every candidate window whose monolayer probability meets the
threshold, extracts a larger region centred on that position directly from the
slide with OpenSlide.

Usage:
    python src/preprocessing/extract_tiles_from_qc.py \
        --qc_dir         data/qc_output/_quality_control \
        --slides_dir     data/slides \
        --out_root       data/tiles \
        --prob_threshold 1.0 \
        --tile_size      1500
"""


import argparse
import re
import sys
from pathlib import Path

# Regex to parse QC log lines
# Matches: 🔎 0: 0,512, prob = 0.0001
# Also handles ASCII fallback: 0: 0,512, prob = 0.0001
QC_LINE = re.compile(r'(\d+):\s*(\d+),(\d+),\s*prob\s*=\s*([0-9.]+)')


def parse_qc_file(qc_path: Path, prob_threshold: float) -> list[tuple[int, int, float]]:
    """
    Parse QC log file and return list of (x, y, prob) for tiles above threshold.
    x, y are the top-left coordinates of the small (512px) QC tile.
    """
    tiles = []
    with open(qc_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = QC_LINE.search(line)
            if not m:
                continue
            x   = int(m.group(2))
            y   = int(m.group(3))
            prob = float(m.group(4))
            if prob >= prob_threshold:
                tiles.append((x, y, prob))
    return tiles


def extract_tiles_for_slide(
    slide_path: Path,
    qc_path: Path,
    out_dir: Path,
    prob_threshold: float,
    tile_size: int,
    small_tile: int = 512,
) -> tuple[int, int]:
    """
    Extract large tiles centred on high-confidence QC tile positions.
    Returns (n_candidates, n_saved).
    """
    try:
        import openslide
    except ImportError:
        print("ERROR: openslide not installed — run: pip install openslide-python")
        sys.exit(1)

    candidates = parse_qc_file(qc_path, prob_threshold)
    if not candidates:
        print(f"  No tiles above threshold {prob_threshold}")
        return 0, 0

    slide = openslide.OpenSlide(str(slide_path))
    W, H  = slide.dimensions
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for x_qc, y_qc, prob in candidates:
        # Centre the large tile on the small QC tile centre
        cx = x_qc + small_tile // 2
        cy = y_qc + small_tile // 2
        x0 = max(0, min(cx - tile_size // 2, W - tile_size))
        y0 = max(0, min(cy - tile_size // 2, H - tile_size))

        tile_name = f"tile_{saved:05d}_prob{prob:.2f}.png"
        out_path  = out_dir / tile_name

        if out_path.exists():
            saved += 1
            continue

        try:
            region = slide.read_region((x0, y0), 0, (tile_size, tile_size)).convert("RGB")
            region.save(str(out_path))
            saved += 1
        except Exception as e:
            print(f"  WARNING: could not extract tile at ({x0},{y0}): {e}")

    slide.close()
    return len(candidates), saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc_dir",     required=True,
                    help="Directory containing per-slide QC files (e.g. _quality_control/)")
    ap.add_argument("--slides_dir", required=True,
                    help="Directory containing MRXS slide files")
    ap.add_argument("--out_root",   required=True,
                    help="Output root — tiles go into <out_root>/<slide_id>/")
    ap.add_argument("--prob_threshold", type=float, default=0.5,
                    help="Minimum QC probability to include a tile (default: 0.5)")
    ap.add_argument("--tile_size",  type=int, default=1500,
                    help="Output tile size in pixels (default: 3000)")
    args = ap.parse_args()

    qc_dir    = Path(args.qc_dir)
    slides_dir = Path(args.slides_dir)
    out_root  = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"QC dir       : {qc_dir}")
    print(f"Slides dir   : {slides_dir}")
    print(f"Output root  : {out_root}")
    print(f"Prob threshold: {args.prob_threshold}")
    print(f"Tile size    : {args.tile_size}px")
    print("=" * 60)

    total_saved   = 0
    missing_slide = 0
    missing_qc    = 0
    skipped       = 0

    for mrxs in SLIDES:
        slide_id  = mrxs.replace(".mrxs", "")
        slide_path = slides_dir / mrxs
        qc_path   = qc_dir / slide_id
        out_dir   = out_root / slide_id

        # Skip if already done
        existing = list(out_dir.glob("*.png")) if out_dir.exists() else []
        if existing:
            print(f"  [SKIP]  {slide_id} ({len(existing)} tiles already exist)")
            total_saved += len(existing)
            skipped += 1
            continue

        if not slide_path.exists():
            print(f"  [WARN]  {slide_id} — MRXS not found")
            missing_slide += 1
            continue

        if not qc_path.exists():
            print(f"  [WARN]  {slide_id} — QC file not found")
            missing_qc += 1
            continue

        print(f"  [PROC]  {slide_id}")
        n_cand, n_saved = extract_tiles_for_slide(
            slide_path, qc_path, out_dir,
            args.prob_threshold, args.tile_size
        )
        print(f"          candidates={n_cand} saved={n_saved}")
        total_saved += n_saved

    print()
    print("=" * 60)
    print(f"Total tiles saved : {total_saved}")
    print(f"Skipped (done)    : {skipped}")
    print(f"Missing MRXS      : {missing_slide}")
    print(f"Missing QC        : {missing_qc}")
    print(f"Output            : {out_root}")


if __name__ == "__main__":
    main()
