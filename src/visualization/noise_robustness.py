#!/usr/bin/env python3
"""
Noise-injection robustness evaluation.

Applies three perturbation types at three severity levels to the RGB channels of
held-out tiles and recomputes discrimination metrics for each condition. The
foreground mask channel is left unperturbed so that any change in performance is
attributable to image quality alone.

  Gaussian noise      sigma  = 10, 25, 50
  Gaussian blur       kernel = 3, 7, 15
  Brightness scaling  factor = 0.6, 0.8, 1.4

Usage:
    python src/visualization/noise_robustness.py \
        --ckpt      outputs/models/cv0/model_best.pth \
        --model     resnet34 \
        --csv       all=data/metadata/dataset_test.csv \
        --csv       male=data/metadata/dataset_test_male.csv \
        --csv       female=data/metadata/dataset_test_female.csv \
        --imgs_root data/tiles \
        --masks_dir data/masks \
        --out_dir   outputs/noise
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
from PIL import Image, ImageFilter
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    roc_auc_score, roc_curve,
)
from torch.utils.data import DataLoader, Dataset

# ── Publication style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#f0f6fc",   # very light blue background
    "axes.facecolor":    "#f7fbff",   # near-white blue tint
    "axes.edgecolor":    "#2c5f8a",
    "axes.linewidth":    1.2,
    "axes.labelcolor":   "#1a3a5c",
    "axes.labelsize":    12,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.titlecolor":   "#1a3a5c",
    "xtick.color":       "#1a3a5c",
    "xtick.labelsize":   10,
    "ytick.color":       "#1a3a5c",
    "ytick.labelsize":   10,
    "text.color":        "#1a3a5c",
    "grid.color":        "#c5ddf0",
    "grid.linestyle":    "--",
    "grid.alpha":        0.7,
    "legend.fontsize":   9,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "#a8c8e8",
    "legend.facecolor":  "#eaf4fb",
    "font.family":       "sans-serif",
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "#f0f6fc",
    "pdf.fonttype":      42,
})

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

RUN_LABELS = {
    "all": "Run 1 — All",
    "male": "Run 2 — Male",
    "female": "Run 3 — Female",
}
RUN_COLORS  = {"all": "#1a6faf", "male": "#4ca3dd", "female": "#a8d4f0"}
RUN_MARKERS = {"all": "o",       "male": "s",       "female": "^"}

# ── Noise levels ───────────────────────────────────────────────────────────────
NOISE_CONFIGS = [
    # (type, severity_label, severity_value)
    ("gaussian",    "Low (σ=10)",    10),
    ("gaussian",    "Med (σ=25)",    25),
    ("gaussian",    "High (σ=50)",   50),
    ("blur",        "Low (k=3)",      3),
    ("blur",        "Med (k=7)",      7),
    ("blur",        "High (k=15)",   15),
    ("brightness",  "Dark (0.6)",    0.6),
    ("brightness",  "Dim (0.8)",     0.8),
    ("brightness",  "Bright (1.4)", 1.4),
]


# ── Noise functions (applied to PIL RGB image) ─────────────────────────────────
def apply_noise(img: Image.Image, noise_type: str, severity) -> Image.Image:
    arr = np.array(img).astype(np.float32)

    if noise_type == "gaussian":
        noise = np.random.normal(0, severity, arr.shape)
        arr   = np.clip(arr + noise, 0, 255)

    elif noise_type == "blur":
        img = img.filter(ImageFilter.GaussianBlur(radius=severity/2))
        return img

    elif noise_type == "brightness":
        arr = np.clip(arr * severity, 0, 255)

    return Image.fromarray(arr.astype(np.uint8))


# ── Dataset ────────────────────────────────────────────────────────────────────
class NoisyTileDataset(Dataset):
    def __init__(self, df, imgs_root, masks_root=None,
                 input_size=(512,512),
                 noise_type=None, severity=None):
        self.samples    = []
        self.input_size = input_size
        self.noise_type = noise_type
        self.severity   = severity

        for _, row in df.iterrows():
            rel   = row["relative_img_path"]
            label = int(row["label"])
            sid   = str(row.get("slide_id",
                        Path(rel).parts[-2] if len(Path(rel).parts)>1 else "unk"))
            img_p = Path(imgs_root) / rel
            mask_p = None
            if masks_root:
                stem  = Path(rel).stem
                mname = stem + "_mask.png"
                for c in [Path(masks_root)/sid/mname,
                          Path(masks_root)/mname]:
                    if c.exists(): mask_p = c; break
            self.samples.append((img_p, mask_p, label, sid))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_p, mask_p, label, sid = self.samples[idx]
        if not img_p.exists():
            return torch.zeros(4, *self.input_size), -1, sid

        img = Image.open(img_p).convert("RGB")
        img = TF.resize(img, list(self.input_size))

        # Apply noise
        if self.noise_type is not None:
            img = apply_noise(img, self.noise_type, self.severity)

        img_t = TF.normalize(TF.to_tensor(img), IMAGENET_MEAN, IMAGENET_STD)

        if mask_p and mask_p.exists():
            mask = TF.to_tensor(TF.resize(
                Image.open(mask_p).convert("L"), list(self.input_size),
                interpolation=TF.InterpolationMode.NEAREST))
        else:
            mask = torch.zeros(1, *self.input_size)

        return torch.cat([img_t, mask], dim=0), label, sid


# ── Model ──────────────────────────────────────────────────────────────────────
def build_model(model_name, ckpt_path, device):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    sd   = ckpt.get("model", ckpt.get("state_dict", ckpt))

    first_key = next((k for k in sd if "weight" in k
                      and len(sd[k].shape)==4), None)
    in_ch = sd[first_key].shape[1] if first_key else 3

    model = tvm.__dict__[model_name](pretrained=False)

    if hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            model.classifier[-1] = nn.Linear(
                model.classifier[-1].in_features, 2)
        else:
            model.classifier = nn.Linear(model.classifier.in_features, 2)

    conv_name, conv = next(
        ((n,m) for n,m in model.named_modules() if isinstance(m,nn.Conv2d)),
        (None,None))

    def set_mod(model, name, mod):
        parts = name.split(".")
        p = model
        for part in parts[:-1]: p = getattr(p,part)
        setattr(p, parts[-1], mod)

    if in_ch == 4:
        new_c = nn.Conv2d(4, conv.out_channels, conv.kernel_size,
                          conv.stride, conv.padding,
                          bias=conv.bias is not None)
        set_mod(model, conv_name, new_c)
        model.load_state_dict(sd, strict=False)
    else:
        model.load_state_dict(sd, strict=False)
        new_c = nn.Conv2d(4, conv.out_channels, conv.kernel_size,
                          conv.stride, conv.padding,
                          bias=conv.bias is not None)
        with torch.no_grad():
            new_c.weight[:,:3] = conv.weight
            new_c.weight[:,3]  = 0.
        set_mod(model, conv_name, new_c)

    return model.to(device).eval()


# ── Inference ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def infer(model, df, imgs_root, masks_dir, input_size,
          batch_size, workers, device, noise_type=None, severity=None):
    ds = NoisyTileDataset(df, imgs_root, masks_dir, input_size,
                          noise_type, severity)
    loader = DataLoader(ds, batch_size=batch_size,
                        num_workers=workers, shuffle=False, pin_memory=True)
    probs, labels = [], []
    for x, y, _ in loader:
        valid = y != -1
        if not valid.any(): continue
        p = F.softmax(model(x[valid].to(device)), dim=1)[:,1].cpu().numpy()
        probs.extend(p)
        labels.extend(y[valid].numpy())
    return np.array(probs), np.array(labels)


# ── Metrics ────────────────────────────────────────────────────────────────────
def quick_metrics(y, p):
    if len(set(y)) < 2:
        return dict(auroc=0.5, ap=0., f1=0., sensitivity=0., specificity=0.)
    fpr,tpr,thr = roc_curve(y,p)
    opt = float(thr[np.argmax(tpr-fpr)])
    yp  = (p>=opt).astype(int)
    from sklearn.metrics import confusion_matrix
    tn,fp,fn,tp = confusion_matrix(y,yp,labels=[0,1]).ravel()
    return dict(
        auroc       = round(roc_auc_score(y,p),4),
        ap          = round(average_precision_score(y,p),4),
        f1          = round(f1_score(y,yp,zero_division=0),4),
        sensitivity = round(tp/(tp+fn) if tp+fn else 0.,4),
        specificity = round(tn/(tn+fp) if tn+fp else 0.,4),
    )


# ── Evaluate one run at all noise levels ───────────────────────────────────────
def evaluate_run(model, df, imgs_root, masks_dir, input_size,
                 batch_size, workers, device, run_name):
    rows = []

    # Baseline (clean)
    print(f"    [clean]")
    p,y = infer(model, df, imgs_root, masks_dir, input_size,
                batch_size, workers, device)
    m = quick_metrics(y,p)
    rows.append({"run":run_name,"noise_type":"clean",
                 "severity_label":"Clean","severity":0,**m})
    baseline_auroc = m["auroc"]
    print(f"      AUROC={m['auroc']}")

    # Noisy passes
    for noise_type, sev_label, severity in NOISE_CONFIGS:
        print(f"    [{noise_type} {sev_label}]")
        p,y = infer(model, df, imgs_root, masks_dir, input_size,
                    batch_size, workers, device, noise_type, severity)
        m = quick_metrics(y,p)
        m["auroc_drop"] = round(baseline_auroc - m["auroc"], 4)
        rows.append({"run":run_name,"noise_type":noise_type,
                     "severity_label":sev_label,"severity":severity,**m})
        print(f"      AUROC={m['auroc']} (drop={m['auroc_drop']:+.4f})")

    return pd.DataFrame(rows)


# ── Figures ────────────────────────────────────────────────────────────────────
def fig_noise_lineplot(results_df, out_path):
    """
    Line plot: AUROC vs noise severity, one line per run, one subplot per noise type.
    """
    noise_types = ["gaussian","blur","brightness"]
    noise_labels = {
        "gaussian":   "Gaussian Noise (σ)",
        "blur":       "Gaussian Blur (kernel size)",
        "brightness": "Brightness Factor",
    }
    runs = sorted(results_df["run"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(15,5), constrained_layout=True)

    for ax, nt in zip(axes, noise_types):
        clean = results_df[results_df["noise_type"]=="clean"]

        for run in runs:
            # Baseline point
            bl = clean[clean["run"]==run]["auroc"].values
            if not len(bl): continue
            baseline = float(bl[0])

            # Noisy points
            sub = results_df[(results_df["run"]==run) &
                             (results_df["noise_type"]==nt)].copy()
            if sub.empty: continue

            # x-axis: severity index (1,2,3)
            x  = [0] + list(range(1, len(sub)+1))
            y  = [baseline] + list(sub["auroc"].values)
            xl = ["Clean"] + list(sub["severity_label"].values)

            ax.plot(x, y,
                    marker=RUN_MARKERS.get(run,"o"),
                    color=RUN_COLORS.get(run,"#333"),
                    lw=2, ms=7,
                    label=RUN_LABELS.get(run,run))

        ax.set_xticks(range(4))
        ax.set_xticklabels(["Clean"]+
            [r["severity_label"] for _,r in
             results_df[results_df["noise_type"]==nt].drop_duplicates(
                 "severity_label").iterrows()],
            rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("AUROC" if ax==axes[0] else "")
        ax.set_title(noise_labels.get(nt,nt))
        ax.set_ylim([max(0, results_df["auroc"].min()-0.05), 1.02])
        ax.grid(True, alpha=0.4)
        ax.axhline(0.5, color="#cccccc", ls=":", lw=1)

    axes[0].legend(loc="lower left")
    fig.suptitle("Noise Robustness — AUROC vs Noise Severity",
                 fontsize=14, fontweight="bold", y=1.02, color="#1a3a5c")

    for ext in [".png",".pdf"]:
        fig.savefig(str(out_path)+ext)
    plt.close(fig)
    print(f"  Saved: {out_path}.png/.pdf")


def fig_noise_heatmap(results_df, out_path):
    """
    Heatmap: rows=noise conditions, cols=runs, values=AUROC drop from baseline.
    """
    runs   = sorted(results_df["run"].unique())
    noisy  = results_df[results_df["noise_type"]!="clean"].copy()
    labels = noisy.drop_duplicates("severity_label")["severity_label"].tolist()

    matrix = np.zeros((len(labels), len(runs)))
    for i,lbl in enumerate(labels):
        for j,run in enumerate(runs):
            val = noisy[(noisy["severity_label"]==lbl) &
                        (noisy["run"]==run)]["auroc_drop"].values
            matrix[i,j] = float(val[0]) if len(val) else 0.

    fig, ax = plt.subplots(figsize=(max(5,len(runs)*2.5), max(5,len(labels)*0.6)))
    # Blue colormap: low drop = light blue, high drop = dark blue
    im = ax.imshow(matrix, cmap="Blues", aspect="auto",
                   vmin=0, vmax=max(0.1, matrix.max()))

    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([RUN_LABELS.get(r,r) for r in runs], fontsize=10)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate cells — white text on dark blue, dark text on light blue
    for i in range(len(labels)):
        for j in range(len(runs)):
            brightness = matrix[i,j] / max(0.001, matrix.max())
            txt_color  = "white" if brightness > 0.55 else "#1a3a5c"
            ax.text(j, i, f"{matrix[i,j]:.3f}",
                    ha="center", va="center", fontsize=9,
                    fontweight="bold", color=txt_color)

    cbar = plt.colorbar(im, ax=ax, label="AUROC Drop from Clean Baseline")
    cbar.ax.yaxis.label.set_color("#1a3a5c")
    cbar.ax.tick_params(colors="#1a3a5c")
    ax.set_title("Noise Robustness Heatmap — AUROC Drop",
                 fontsize=13, fontweight="bold", color="#1a3a5c")
    fig.tight_layout()

    for ext in [".png",".pdf"]:
        fig.savefig(str(out_path)+ext)
    plt.close(fig)
    print(f"  Saved: {out_path}.png/.pdf")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",       required=True)
    ap.add_argument("--model",      default="resnet34")
    ap.add_argument("--csv", action="append", default=[], metavar="NAME=PATH",
                    help="Cohort test CSV given as NAME=PATH. Repeatable, e.g. "
                         "--csv all=test.csv --csv male=test_male.csv")
    ap.add_argument("--imgs_root",  required=True)
    ap.add_argument("--masks_dir",  default=None)
    ap.add_argument("--out_dir",    required=True)
    ap.add_argument("--input_size", type=int, nargs=2, default=[512,512])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers",    type=int, default=4)
    ap.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device(args.device)

    print(f"\n{'='*60}")
    print("Noise Robustness Evaluation")
    print(f"Model : {args.model} | {args.ckpt}")
    print(f"Device: {device}")
    print(f"Noise : {len(NOISE_CONFIGS)} configs (3 types × 3 levels) + clean")
    print(f"{'='*60}")

    # Load model
    print(f"\n── Loading model")
    model = build_model(args.model, Path(args.ckpt), device)
    print(f"  Loaded ✓")

    run_csvs = {}
    for spec in args.csv:
        if "=" not in spec:
            sys.exit(f"--csv expects NAME=PATH, got: {spec}")
        name, path = spec.split("=", 1)
        run_csvs[name] = path
    if not run_csvs:
        sys.exit("No cohort CSVs provided. Use --csv NAME=PATH (repeatable).")

    all_results = []

    for run, csv_path in run_csvs.items():
        if not csv_path or not Path(csv_path).exists():
            print(f"\n  {run}: CSV not provided — skipping")
            continue

        df = pd.read_csv(csv_path)
        print(f"\n── {RUN_LABELS.get(run,run)} ({len(df)} tiles)")

        run_results = evaluate_run(
            model, df, args.imgs_root, args.masks_dir,
            tuple(args.input_size), args.batch_size,
            args.workers, device, run
        )
        all_results.append(run_results)
        run_results.to_csv(out_dir/f"noise_results_{run}.csv", index=False)

    if not all_results:
        print("No results generated — check CSV paths")
        sys.exit(1)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(out_dir/"noise_results_all.csv", index=False)
    print(f"\n  Saved: noise_results_all.csv")

    # Figures
    print(f"\n── Generating figures")
    fig_noise_lineplot(combined, out_dir/"noise_robustness_lineplot")
    fig_noise_heatmap(combined,  out_dir/"noise_robustness_heatmap")

    # Print summary
    print(f"\n{'='*60}")
    print("Summary — AUROC Drop (clean → highest severity):")
    for run in run_csvs:
        sub = combined[combined["run"]==run]
        if sub.empty: continue
        clean = sub[sub["noise_type"]=="clean"]["auroc"].values
        if not len(clean): continue
        max_drop = sub[sub["noise_type"]!="clean"]["auroc_drop"].max()
        print(f"  {RUN_LABELS.get(run,run):30s}: "
              f"baseline={clean[0]:.4f}  max_drop={max_drop:+.4f}")

    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
