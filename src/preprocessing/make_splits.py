#!/usr/bin/env python3
"""
Build stratified group k-fold train/validation/test splits.

Groups tiles by slide identifier so that no slide contributes tiles to more than
one fold. One fold is held out as the test set; the remainder are used for
cross-validation.

Input CSV columns: relative_img_path, label, slide_id

Usage:
    python src/preprocessing/make_splits.py \
        --csv       data/metadata/dataset.csv \
        --out_dir   data/metadata \
        --prefix    dataset \
        --n_folds   5 \
        --test_fold 4
"""


import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def make_split(csv_path: Path, out_dir: Path, prefix: str,
               n_folds: int, test_fold: int, seed: int) -> None:

    df = pd.read_csv(csv_path)
    assert "relative_img_path" in df.columns, "CSV must have 'relative_img_path' column"
    assert "label" in df.columns,             "CSV must have 'label' column"
    assert "slide_id" in df.columns,          "CSV must have 'slide_id' column"

    print(f"\n{'='*60}")
    print(f"Stratified Group {n_folds}-Fold Split")
    print(f"Test fold: {test_fold}  |  random_state: {seed}")
    print(f"{'='*60}")

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                                random_state=seed)

    y = df["label"].values
    g = df["slide_id"].values
    splits = list(sgkf.split(df, y, g))

    test_idx  = splits[test_fold][1]
    train_idx = np.concatenate([splits[i][1]
                                for i in range(n_folds) if i != test_fold])

    trainval = df.iloc[train_idx].reset_index(drop=True)
    test     = df.iloc[test_idx].reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    trainval_path = out_dir / f"{prefix}_trainval.csv"
    test_path     = out_dir / f"{prefix}_test.csv"
    trainval.to_csv(trainval_path, index=False)
    test.to_csv(test_path, index=False)

    n_slides_tv  = trainval["slide_id"].nunique()
    n_slides_test = test["slide_id"].nunique()
    ctrl_tv  = int((trainval.label==0).sum())
    epo_tv   = int((trainval.label==1).sum())
    ctrl_te  = int((test.label==0).sum())
    epo_te   = int((test.label==1).sum())

    print(f"  Total tiles  : {len(df):,}")
    print(f"  Trainval     : {len(trainval):,} tiles | "
          f"{n_slides_tv} slides | ctrl={ctrl_tv} epo={epo_tv}")
    print(f"  Test         : {len(test):,} tiles | "
          f"{n_slides_test} slides | ctrl={ctrl_te} epo={epo_te}")

    ratio_tv = epo_tv / ctrl_tv if ctrl_tv else float("inf")
    ratio_te = epo_te / ctrl_te if ctrl_te else float("inf")
    print(f"  EPO:ctrl ratio — trainval={ratio_tv:.2f}  test={ratio_te:.2f}")
    if abs(ratio_te - ratio_tv) > 1.0:
        print(f"  ⚠  Class imbalance difference between splits — "
              f"check cohort composition")

    print(f"  Trainval CSV : {trainval_path}")
    print(f"  Test CSV     : {test_path}")
    print("[DONE] Splits written.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",       required=True,
                    help="Input tile CSV (relative_img_path, label, slide_id)")
    ap.add_argument("--out_dir",   required=True,
                    help="Directory to write trainval/test CSVs")
    ap.add_argument("--prefix",    default="dataset",
                    help="Filename prefix for output CSVs")
    ap.add_argument("--n_folds",   type=int, default=5)
    ap.add_argument("--test_fold", type=int, default=4,
                    help="Which fold to hold out as test set")
    ap.add_argument("--seed",      type=int, default=42)
    args = ap.parse_args()

    make_split(Path(args.csv), Path(args.out_dir),
               args.prefix, args.n_folds, args.test_fold, args.seed)


if __name__ == "__main__":
    main()
