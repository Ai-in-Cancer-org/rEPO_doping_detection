#!/usr/bin/env python3
"""
Inference and performance evaluation for 4-channel CNN classifiers.

Computes tile-level probabilities, aggregates them to slide level by averaging
the softmax output across all tiles of a slide, and reports discrimination and
calibration metrics at both a fixed threshold and the Youden-optimal threshold.

Outputs:
  tile_predictions.csv, slide_predictions.csv, metrics_summary.csv,
  ROC and precision-recall curves, confusion matrices, confidence histogram.

Robustness under image degradation is evaluated separately by
src/visualization/noise_robustness.py.

Usage:
    python src/evaluation/inference.py \
        --ckpt      outputs/models/cv0/model_best.pth \
        --model     resnet34 \
        --csv       data/metadata/dataset_test.csv \
        --imgs_root data/tiles \
        --masks_dir data/masks \
        --out_dir   outputs/results
"""


import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_recall_curve, roc_auc_score, roc_curve,
)
from torch.utils.data import DataLoader, Dataset

# ── Plot style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor":  "white",
    "axes.edgecolor":   "#333",  "axes.labelsize":  12,
    "axes.titlesize":   13,      "axes.titleweight": "bold",
    "font.family":      "sans-serif", "figure.dpi": 150,
    "savefig.dpi":      300,     "savefig.bbox":    "tight",
    "pdf.fonttype":     42,
})

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Dataset ────────────────────────────────────────────────────────────────────
class CSVInferenceDataset(Dataset):
    """Load tiles from CSV: relative_img_path, label, slide_id."""
    def __init__(self, csv_path, imgs_root, masks_root=None,
                 input_size=(512, 512)):
        self.input_size = input_size
        self.samples = []
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            rel   = row["relative_img_path"]
            label = int(row["label"])
            sid   = str(row.get("slide_id",
                        Path(rel).parts[-2] if len(Path(rel).parts) > 1
                        else "unknown"))
            img_p  = Path(imgs_root) / rel
            mask_p = self._find_mask(img_p, sid, masks_root)
            self.samples.append((img_p, mask_p, label, sid, rel))

        n1 = sum(1 for *_, l, _, _ in self.samples if l == 1)
        n0 = sum(1 for *_, l, _, _ in self.samples if l == 0)
        missing = sum(1 for _, m, *_ in self.samples if m is None)
        print(f"  Dataset: {len(self.samples)} tiles | "
              f"label=1: {n1} | label=0: {n0}")
        if missing:
            print(f"  WARNING: {missing} tiles without mask — using blank")

    @staticmethod
    def _find_mask(img_path, slide_id, masks_root):
        if masks_root is None:
            return None
        stem  = img_path.stem
        mname = stem + "_mask.png"
        for c in [Path(masks_root)/slide_id/mname,
                  Path(masks_root)/mname]:
            if c.exists():
                return c
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, mask_p, label, sid, rel = self.samples[idx]
        if not img_p.exists():
            return torch.zeros(4, *self.input_size), -1, rel, sid
        img = TF.normalize(
            TF.to_tensor(TF.resize(Image.open(img_p).convert("RGB"),
                                    list(self.input_size))),
            IMAGENET_MEAN, IMAGENET_STD)
        if mask_p and mask_p.exists():
            mask = TF.to_tensor(TF.resize(
                Image.open(mask_p).convert("L"), list(self.input_size),
                interpolation=TF.InterpolationMode.NEAREST))
        else:
            mask = torch.zeros(1, *self.input_size)
        return torch.cat([img, mask], dim=0), label, rel, sid


class FolderInferenceDataset(Dataset):
    """
    Load tiles from folder structure:
        root/positive/  → label=1
        root/negative/  → label=0
    Positive/negative folder names are configurable via --pos_dir / --neg_dir.
    """
    def __init__(self, root, masks_root=None, input_size=(512, 512),
                 pos_dir="positive", neg_dir="negative"):
        self.input_size = input_size
        self.samples = []
        root = Path(root)
        for label, folder in [(1, pos_dir), (0, neg_dir)]:
            d = root / folder
            if not d.exists():
                print(f"  WARNING: {d} not found")
                continue
            for img_p in sorted(d.glob("*.png")):
                sid   = img_p.stem.split("_tile_")[0].replace(".", "p")
                mask_p = CSVInferenceDataset._find_mask(
                    img_p, sid, masks_root)
                self.samples.append((img_p, mask_p, label, sid,
                                     str(img_p)))

        n1 = sum(1 for *_, l, _, _ in self.samples if l == 1)
        n0 = sum(1 for *_, l, _, _ in self.samples if l == 0)
        print(f"  Dataset: {len(self.samples)} tiles | "
              f"label=1 ({pos_dir}): {n1} | label=0 ({neg_dir}): {n0}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        return CSVInferenceDataset.__getitem__(self, idx)


# ── Model ──────────────────────────────────────────────────────────────────────
def build_model(model_name: str, ckpt_path: Path, device: torch.device,
                n_classes: int = 2):
    """
    Load a checkpoint into a torchvision model.
    Automatically detects whether the checkpoint used 3 or 4 input channels
    and patches the first conv layer accordingly.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    sd   = ckpt.get("model", ckpt.get("state_dict", ckpt))

    # Detect input channels from checkpoint
    first_key = next((k for k in sd
                      if "weight" in k and len(sd[k].shape) == 4), None)
    in_ch = sd[first_key].shape[1] if first_key else 3
    print(f"  Checkpoint: {in_ch}-channel input detected")

    model = tvm.__dict__[model_name](weights=None)

    # Patch classifier head
    if hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            model.classifier[-1] = nn.Linear(
                model.classifier[-1].in_features, n_classes)
        else:
            model.classifier = nn.Linear(
                model.classifier.in_features, n_classes)

    # Find and patch first conv
    conv_name, conv = next(
        ((n, m) for n, m in model.named_modules()
         if isinstance(m, nn.Conv2d)),
        (None, None))

    def _set(model, name, mod):
        parts = name.split(".")
        p = model
        for part in parts[:-1]: p = getattr(p, part)
        setattr(p, parts[-1], mod)

    if in_ch == 4:
        new_c = nn.Conv2d(4, conv.out_channels, conv.kernel_size,
                          conv.stride, conv.padding,
                          bias=conv.bias is not None)
        _set(model, conv_name, new_c)
        model.load_state_dict(sd, strict=False)
    else:
        model.load_state_dict(sd, strict=False)
        new_c = nn.Conv2d(4, conv.out_channels, conv.kernel_size,
                          conv.stride, conv.padding,
                          bias=conv.bias is not None)
        with torch.no_grad():
            new_c.weight[:, :3] = conv.weight
            new_c.weight[:, 3]  = 0.0
        _set(model, conv_name, new_c)

    return model.to(device).eval()


# ── Inference ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, loader, device):
    probs, labels, rels, slides = [], [], [], []
    skipped = 0
    for x, y, r, s in loader:
        valid = y != -1
        skipped += int((~valid).sum())
        if not valid.any():
            continue
        p = F.softmax(model(x[valid].to(device)), dim=1)[:, 1].cpu().numpy()
        probs.extend(p)
        labels.extend(y[valid].numpy())
        rels.extend([r[i] for i in range(len(r)) if valid[i]])
        slides.extend([s[i] for i in range(len(s)) if valid[i]])
    if skipped:
        print(f"  Skipped {skipped} tiles (file not found on disk)")
    return np.array(probs), np.array(labels), rels, slides


# ── Metrics ────────────────────────────────────────────────────────────────────
def youden_threshold(fpr, tpr, thresholds):
    return float(thresholds[np.argmax(tpr - fpr)])


def compute_metrics(y, p, threshold, n_bins=10, label=""):
    yp = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    auroc = roc_auc_score(y, p) if len(set(y)) > 1 else 0.5
    ap    = average_precision_score(y, p) if len(set(y)) > 1 else 0.0
    # ECE
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        ece += m.sum() * abs(p[m].mean() - y[m].mean())
    ece /= max(len(y), 1)
    return dict(
        label=label,
        auroc=round(auroc, 4), ap=round(ap, 4),
        f1=round(f1_score(y, yp, zero_division=0), 4),
        accuracy=round(accuracy_score(y, yp), 4),
        sensitivity=round(tp/(tp+fn) if tp+fn else 0., 4),
        specificity=round(tn/(tn+fp) if tn+fp else 0., 4),
        ppv=round(tp/(tp+fp) if tp+fp else 0., 4),
        npv=round(tn/(tn+fn) if tn+fn else 0., 4),
        ece=round(ece, 4),
        threshold=round(threshold, 4),
        tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn), n=len(y),
    )


# ── Figures ────────────────────────────────────────────────────────────────────
def save_confusion_matrix(y, yp, title, path,
                           class_names=("negative", "positive")):
    cm = confusion_matrix(y, yp, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="white" if cm[i, j] > cm.max()/2 else "#333")
    plt.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def save_roc(y, p, title, path, opt_thr=None):
    fpr, tpr, thr = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    thr_padded = np.pad(thr, (0, len(fpr)-len(thr)), constant_values=np.nan)
    pd.DataFrame({"fpr": fpr, "tpr": tpr,
                  "threshold": thr_padded,
                  "auroc": round(auc, 4)}).to_csv(
        str(path).replace(".png", "_data.csv"), index=False)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, color="#1a5fa8", label=f"AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="#aaa", lw=0.8)
    if opt_thr:
        idx = np.argmin(np.abs(thr - opt_thr))
        ax.plot(fpr[idx], tpr[idx], "o", color="#e07020", ms=8,
                label=f"Youden t={opt_thr:.3f}")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(title)
    ax.legend(); ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def save_pr(y, p, title, path):
    prec, rec, thr = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    thr_padded = np.pad(thr, (0, len(prec)-len(thr)), constant_values=np.nan)
    pd.DataFrame({"precision": prec, "recall": rec,
                  "threshold": thr_padded,
                  "ap": round(ap, 4)}).to_csv(
        str(path).replace(".png", "_data.csv"), index=False)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rec, prec, lw=2, color="#228833", label=f"AP={ap:.4f}")
    ax.axhline(float(y.mean()), color="#aaa", ls="--", lw=0.8,
               label=f"Baseline={y.mean():.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(title)
    ax.legend(); ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    fig.tight_layout(); fig.savefig(path); plt.close(fig)



def save_conf_hist(y, p, title, path, class_names=("negative", "positive")):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(p[y==0], bins=30, alpha=0.6, color="#1a5fa8",
            label=class_names[0], density=True)
    ax.hist(p[y==1], bins=30, alpha=0.6, color="#e07020",
            label=class_names[1], density=True)
    ax.set_xlabel("Predicted probability (positive class)")
    ax.set_ylabel("Density"); ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Standalone inference for 4-channel ResNet models")
    ap.add_argument("--ckpt",       required=True,
                    help="Path to model checkpoint (.pth)")
    ap.add_argument("--model",      default="resnet34",
                    help="Model architecture (e.g. resnet34, resnet50, "
                         "efficientnet_b2)")
    ap.add_argument("--n_classes",  type=int, default=2)
    ap.add_argument("--n_bins",     type=int, default=10,
                    help="Bins used for the expected calibration error value")

    # Input mode A: CSV
    ap.add_argument("--csv",        default=None,
                    help="Test CSV (relative_img_path, label, slide_id)")
    ap.add_argument("--imgs_root",  default=None,
                    help="Root directory for images when using --csv")

    # Input mode B: folder
    ap.add_argument("--test_dir",   default=None,
                    help="Folder with positive/ and negative/ subfolders")
    ap.add_argument("--pos_dir",    default="positive",
                    help="Name of positive class subfolder")
    ap.add_argument("--neg_dir",    default="negative",
                    help="Name of negative class subfolder")

    # Shared options
    ap.add_argument("--masks_dir",  default=None,
                    help="Directory containing foreground masks (optional)")
    ap.add_argument("--out_dir",    required=True)
    ap.add_argument("--run_name",   default="inference")
    ap.add_argument("--class_names",nargs=2, default=["negative","positive"],
                    help="Class names for plots (label=0 label=1)")
    ap.add_argument("--input_size", type=int, nargs=2, default=[512, 512])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers",    type=int, default=4)
    ap.add_argument("--device",     default="cuda"
                    if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"Inference — {args.run_name}")
    print(f"Model     : {args.model}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Device    : {device}")
    print(f"{'='*60}")

    # ── Build dataset ─────────────────────────────────────────────────────────
    print(f"\n── Dataset")
    if args.csv:
        if not args.imgs_root:
            print("ERROR: --imgs_root required with --csv"); sys.exit(1)
        ds = CSVInferenceDataset(args.csv, args.imgs_root, args.masks_dir,
                                  tuple(args.input_size))
    elif args.test_dir:
        ds = FolderInferenceDataset(args.test_dir, args.masks_dir,
                                     tuple(args.input_size),
                                     args.pos_dir, args.neg_dir)
    else:
        print("ERROR: provide --csv or --test_dir"); sys.exit(1)

    loader = DataLoader(ds, batch_size=args.batch_size,
                        num_workers=args.workers, shuffle=False,
                        pin_memory=True)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n── Model")
    model = build_model(args.model, Path(args.ckpt), device, args.n_classes)
    print(f"  Loaded ✓")

    # ── Inference ─────────────────────────────────────────────────────────────
    print(f"\n── Running inference")
    probs, labels, rels, slides = run_inference(model, loader, device)
    print(f"  Done — {len(probs)} tiles")

    # Save tile predictions
    tile_df = pd.DataFrame({
        "relative_img_path": rels,
        "slide_id":          slides,
        "label":             labels,
        "prob_positive":     probs,
        "pred_t05":          (probs >= 0.5).astype(int),
    })
    tile_df.to_csv(out_dir/"tile_predictions.csv", index=False)

    # ── Thresholds ────────────────────────────────────────────────────────────
    fpr, tpr, thr = roc_curve(labels, probs)
    opt_thr = youden_threshold(fpr, tpr, thr)
    print(f"\n── Thresholds — fixed: 0.5 | Youden: {opt_thr:.4f}")

    # ── Tile-level metrics ────────────────────────────────────────────────────
    print(f"\n── Tile-level metrics")
    m05  = compute_metrics(labels, probs, 0.5,    args.n_bins, "tile_t0.5")
    mopt = compute_metrics(labels, probs, opt_thr, args.n_bins, "tile_t_opt")
    for tag, m in [("t=0.5", m05), ("Youden", mopt)]:
        print(f"  [{tag}] AUROC={m['auroc']}  F1={m['f1']}  "
              f"Sens={m['sensitivity']}  Spec={m['specificity']}  "
              f"PPV={m['ppv']}  NPV={m['npv']}  ECE={m['ece']}")

    # ── Slide-level ───────────────────────────────────────────────────────────
    print(f"\n── Slide-level aggregation")
    slide_df = (tile_df.groupby("slide_id")
                .agg(mean_prob=("prob_positive","mean"),
                     label=("label","first"),
                     n_tiles=("prob_positive","count"))
                .reset_index())
    slide_df["pred_t05"]  = (slide_df["mean_prob"] >= 0.5).astype(int)
    slide_df["pred_t_opt"]= (slide_df["mean_prob"] >= opt_thr).astype(int)
    slide_df.to_csv(out_dir/"slide_predictions.csv", index=False)
    print(f"  {len(slide_df)} slides")

    all_metrics = [m05, mopt]
    sy, sp = slide_df["label"].values, slide_df["mean_prob"].values
    if len(set(sy)) > 1:
        sfpr, stpr, sthr = roc_curve(sy, sp)
        sopt = youden_threshold(sfpr, stpr, sthr)
        sm05  = compute_metrics(sy, sp, 0.5,  args.n_bins, "slide_t0.5")
        smopt = compute_metrics(sy, sp, sopt, args.n_bins, "slide_t_opt")
        all_metrics += [sm05, smopt]
        print(f"  [slide t=0.5] AUROC={sm05['auroc']}  "
              f"Sens={sm05['sensitivity']}  Spec={sm05['specificity']}")

    pd.DataFrame(all_metrics).to_csv(out_dir/"metrics_summary.csv",
                                      index=False)

    # ── Figures ───────────────────────────────────────────────────────────────
    print(f"\n── Saving figures")
    cn = args.class_names

    save_confusion_matrix(labels, (probs>=0.5).astype(int),
        f"{args.run_name} — Tile CM (t=0.5)",
        out_dir/"confusion_matrix_tile_t05.png", cn)
    save_confusion_matrix(labels, (probs>=opt_thr).astype(int),
        f"{args.run_name} — Tile CM (Youden t={opt_thr:.3f})",
        out_dir/"confusion_matrix_tile_opt.png", cn)

    save_roc(labels, probs,
             f"{args.run_name} — ROC (tile)",
             out_dir/"roc_tile.png", opt_thr)
    save_pr(labels, probs,
            f"{args.run_name} — PR (tile)",
            out_dir/"pr_tile.png")
    save_conf_hist(labels, probs,
                   f"{args.run_name} — Confidence Distribution",
                   out_dir/"confidence_histogram.png", cn)

    if len(set(sy)) > 1:
        save_confusion_matrix(sy, (sp>=sopt).astype(int),
            f"{args.run_name} — Slide CM (Youden t={sopt:.3f})",
            out_dir/"confusion_matrix_slide_opt.png", cn)
        save_roc(sy, sp,
                 f"{args.run_name} — ROC (slide)",
                 out_dir/"roc_slide.png", sopt)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"DONE — {args.run_name}")
    print(f"  Tiles : {len(probs)} | Slides: {len(slide_df)}")
    print(f"  Tile AUROC (t=0.5): {m05['auroc']} | "
          f"Youden: {mopt['auroc']}")
    print(f"  Output: {out_dir}")


if __name__ == "__main__":
    main()
