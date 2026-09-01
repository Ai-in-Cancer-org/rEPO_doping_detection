#!/usr/bin/env python3
"""
Compute tile-level and slide-level metrics for user-defined subgroups.

Subgroups are declared in a YAML file by explicit slide identifier or by slide
identifier prefix; no cohort membership is hardcoded.

Usage:
    python src/evaluation/subgroup_metrics.py \
        --preds       outputs/results/tile_predictions.csv \
        --groups_yaml configs/subgroups.yaml \
        --out_dir     outputs/results/subgroups
"""


import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_recall_curve, roc_auc_score, roc_curve,
)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor":  "white",
    "axes.edgecolor":   "#333",  "axes.labelsize":  11,
    "axes.titlesize":   12,      "axes.titleweight": "bold",
    "font.family":      "sans-serif",
    "figure.dpi": 150,  "savefig.dpi": 300,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})


# ── Metrics ────────────────────────────────────────────────────────────────────
def youden(fpr, tpr, thr):
    return float(thr[np.argmax(tpr - fpr)])


def compute_metrics(y, p, t, label=""):
    yp = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    auroc = roc_auc_score(y, p) if len(set(y)) > 1 else 0.5
    ap    = average_precision_score(y, p) if len(set(y)) > 1 else 0.0
    return dict(
        label=label,
        auroc=round(auroc, 4),
        ap=round(ap, 4),
        f1=round(f1_score(y, yp, zero_division=0), 4),
        accuracy=round(accuracy_score(y, yp), 4),
        sensitivity=round(tp/(tp+fn) if tp+fn else 0., 4),
        specificity=round(tn/(tn+fp) if tn+fp else 0., 4),
        ppv=round(tp/(tp+fp) if tp+fp else 0., 4),
        npv=round(tn/(tn+fn) if tn+fn else 0., 4),
        threshold=round(t, 4),
        tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn), n=len(y),
    )


# ── Figures ────────────────────────────────────────────────────────────────────
def save_roc(y, p, title, path, opt_thr=None):
    if len(set(y)) < 2:
        return
    fpr, tpr, thr = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    thr_pad = np.pad(thr, (0, len(fpr)-len(thr)), constant_values=np.nan)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr_pad,
                  "auroc": round(auc,4)}).to_csv(
        str(path).replace(".png", "_data.csv"), index=False)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, color="#1a5fa8", label=f"AUC={auc:.4f}")
    ax.plot([0,1],[0,1],"--",color="#aaa",lw=0.8)
    if opt_thr:
        idx = np.argmin(np.abs(thr-opt_thr))
        ax.plot(fpr[idx], tpr[idx], "o", color="#e07020", ms=8,
                label=f"Youden t={opt_thr:.3f}")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(title)
    ax.legend(fontsize=8); ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    ax.grid(True, alpha=0.4)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def save_cm(y, yp, title, path):
    cm = confusion_matrix(y, yp, labels=[0,1])
    row_t = cm.sum(axis=1, keepdims=True)
    pct   = 100.0 * cm / np.where(row_t==0, 1, row_t)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            c = "white" if cm[i,j]>thresh else "#333"
            ax.text(j, i-0.12, str(cm[i,j]),
                    ha="center", va="center",
                    fontsize=18, fontweight="bold", color=c)
            ax.text(j, i+0.22, f"({pct[i,j]:.1f}%)",
                    ha="center", va="center", fontsize=10, color=c)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Negative","Positive"])
    ax.set_yticklabels(["Negative","Positive"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ── Subgroup evaluation ────────────────────────────────────────────────────────
def filter_subgroup(df, group_cfg):
    """
    Filter DataFrame to slides matching either slide_ids or slide_id_prefixes.
    """
    if "slide_ids" in group_cfg:
        ids = set(str(s) for s in group_cfg["slide_ids"])
        mask = df["slide_id"].astype(str).isin(ids)
    elif "slide_id_prefixes" in group_cfg:
        prefixes = tuple(str(p) for p in group_cfg["slide_id_prefixes"])
        mask = df["slide_id"].astype(str).str.startswith(prefixes)
    else:
        raise ValueError("Each subgroup must have 'slide_ids' or "
                         "'slide_id_prefixes'")
    return df[mask].copy()


def evaluate_subgroup(df, name, out_dir):
    """Run tile + slide metrics for a subgroup and save all outputs."""
    d = out_dir / name.replace(" ", "_").lower()
    d.mkdir(parents=True, exist_ok=True)

    y = df["label"].values
    p = df["prob_positive"].values if "prob_positive" in df.columns \
        else df["prob_repo"].values

    n1 = int((y==1).sum()); n0 = int((y==0).sum())
    print(f"  {name}: {len(df)} tiles | pos={n1} neg={n0} "
          f"| {df['slide_id'].nunique()} slides")

    if len(set(y)) < 2:
        print(f"  SKIP — only one class in subgroup")
        return None

    # Tile-level
    fpr, tpr, thr = roc_curve(y, p)
    opt = youden(fpr, tpr, thr)
    m_tile_opt = compute_metrics(y, p, opt,  f"{name}_tile_opt")
    m_tile_05  = compute_metrics(y, p, 0.5,  f"{name}_tile_t05")

    df.to_csv(d/"tile_predictions.csv", index=False)
    save_roc(y, p, f"{name} — ROC (tile)", d/"roc_tile.png", opt)
    save_cm(y, (p>=opt).astype(int),
            f"{name} — CM tile (t={opt:.3f})",
            d/"confusion_matrix_tile.png")

    # Slide-level
    sdf = (df.groupby("slide_id")
             .agg(mean_prob=(df.columns[df.columns.str.contains("prob")][0],
                             "mean"),
                  label=("label","first"),
                  n_tiles=(df.columns[0],"count"))
             .reset_index())
    sdf.columns = ["slide_id","mean_prob","label","n_tiles"]
    sdf.to_csv(d/"slide_predictions.csv", index=False)

    m_slide_opt = m_slide_05 = None
    sy, sp = sdf["label"].values, sdf["mean_prob"].values
    if len(set(sy)) > 1:
        sfpr, stpr, sthr = roc_curve(sy, sp)
        sopt = youden(sfpr, stpr, sthr)
        m_slide_opt = compute_metrics(sy, sp, sopt, f"{name}_slide_opt")
        m_slide_05  = compute_metrics(sy, sp, 0.5,  f"{name}_slide_t05")
        save_roc(sy, sp, f"{name} — ROC (slide)", d/"roc_slide.png", sopt)
        save_cm(sy, (sp>=sopt).astype(int),
                f"{name} — CM slide (t={sopt:.3f})",
                d/"confusion_matrix_slide.png")

    # Save metrics
    rows = [m_tile_05, m_tile_opt]
    if m_slide_05:
        rows += [m_slide_05, m_slide_opt]
    pd.DataFrame(rows).to_csv(d/"metrics.csv", index=False)

    print(f"    tile AUROC={m_tile_opt['auroc']}  "
          f"Sens={m_tile_opt['sensitivity']}  "
          f"Spec={m_tile_opt['specificity']}")
    if m_slide_opt:
        print(f"    slide AUROC={m_slide_opt['auroc']}  "
              f"Sens={m_slide_opt['sensitivity']}  "
              f"Spec={m_slide_opt['specificity']}")

    return {
        "subgroup":         name,
        "n_tiles":          len(df),
        "n_slides":         df["slide_id"].nunique(),
        "n_positive":       n1,
        "n_negative":       n0,
        "tile_auroc":       m_tile_opt["auroc"],
        "tile_ap":          m_tile_opt["ap"],
        "tile_f1":          m_tile_opt["f1"],
        "tile_sensitivity": m_tile_opt["sensitivity"],
        "tile_specificity": m_tile_opt["specificity"],
        "tile_ppv":         m_tile_opt["ppv"],
        "tile_npv":         m_tile_opt["npv"],
        "tile_threshold":   m_tile_opt["threshold"],
        "slide_auroc":      m_slide_opt["auroc"] if m_slide_opt else None,
        "slide_sensitivity":m_slide_opt["sensitivity"] if m_slide_opt else None,
        "slide_specificity":m_slide_opt["specificity"] if m_slide_opt else None,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Subgroup metric analysis from tile predictions")
    ap.add_argument("--preds",       required=True,
                    help="tile_predictions.csv from inference.py")
    ap.add_argument("--groups_yaml", required=True,
                    help="YAML file defining subgroups "
                         "(see configs/subgroups_example.yaml)")
    ap.add_argument("--out_dir",     required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Subgroup Metrics Analysis")
    print(f"Predictions : {args.preds}")
    print(f"Groups YAML : {args.groups_yaml}")
    print(f"Output      : {out_dir}")
    print(f"{'='*60}")

    # Load predictions
    df = pd.read_csv(args.preds)
    print(f"\nLoaded {len(df):,} tiles | "
          f"{df['slide_id'].nunique()} slides")

    # Load group definitions
    with open(args.groups_yaml) as f:
        cfg = yaml.safe_load(f)

    subgroups = cfg.get("subgroups", {})
    if not subgroups:
        print("ERROR: no subgroups defined in YAML")
        return

    print(f"Subgroups defined: {list(subgroups.keys())}")

    # Evaluate each subgroup
    summary_rows = []
    for name, group_cfg in subgroups.items():
        print(f"\n── {name}")
        if "description" in group_cfg:
            print(f"   {group_cfg['description']}")
        try:
            sub_df = filter_subgroup(df, group_cfg)
            if sub_df.empty:
                print(f"  WARNING: no matching tiles found for '{name}'")
                continue
            row = evaluate_subgroup(sub_df, name, out_dir)
            if row:
                summary_rows.append(row)
        except Exception as e:
            print(f"  ERROR in subgroup '{name}': {e}")

    if not summary_rows:
        print("\nNo subgroup results generated.")
        return

    # Save summary
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir/"subgroup_summary.csv", index=False)
    print(f"\n{'='*60}")
    print(summary[["subgroup","tile_auroc","tile_sensitivity",
                   "tile_specificity","slide_auroc"]].to_string(index=False))
    print(f"\nSaved: {out_dir}/subgroup_summary.csv")

    try:
        summary.to_excel(out_dir/"subgroup_summary.xlsx", index=False)
        print(f"Saved: {out_dir}/subgroup_summary.xlsx")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
