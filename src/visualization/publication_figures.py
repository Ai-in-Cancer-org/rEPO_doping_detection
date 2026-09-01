#!/usr/bin/env python3
"""
Generate publication figures from cross-validated evaluation output.

Produces per-cohort metric tables, ROC and precision-recall panels showing all
folds with the mean and standard deviation, noise-injection robustness curves,
and pooled confusion matrices.

Usage:
    python src/visualization/publication_figures.py \
        --results_dir outputs/results \
        --noise_csv   outputs/noise/noise_results_all.csv \
        --out_dir     outputs/figures
"""


import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_recall_curve, roc_auc_score, roc_curve,
)

# ── Publication style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#333333",
    "axes.linewidth":    1.2,
    "axes.labelcolor":   "#111111",
    "axes.labelsize":    12,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "xtick.color":       "#333333",
    "ytick.color":       "#333333",
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "grid.color":        "#dddddd",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "legend.fontsize":   9,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "#cccccc",
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica","Arial","DejaVu Sans"],
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
    "pdf.fonttype":      42,   # embeds fonts in PDF
    "ps.fonttype":       42,
})

# Colour palette — 5 folds + mean
FOLD_COLORS  = ["#4477AA","#EE6677","#228833","#CCBB44","#AA3377"]
MEAN_COLOR   = "#000000"
FILL_ALPHA   = 0.12
RUN_LABELS   = {"all": "All participants",
                "male": "Male participants",
                "female": "Female participants"}
RUN_COLORS   = {"all": "#4477AA", "male": "#EE6677", "female": "#228833"}


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_run_data(results_dir: Path, run: str):
    """Load per-fold predictions and metrics for one run."""
    d = results_dir / run
    preds   = pd.read_csv(d/"all_fold_predictions.csv")  if (d/"all_fold_predictions.csv").exists() else None
    metrics = pd.read_csv(d/"per_fold_metrics.csv")      if (d/"per_fold_metrics.csv").exists() else None
    return preds, metrics


def mean_std(arr): return float(np.mean(arr)), float(np.std(arr))


# ── ROC helpers ────────────────────────────────────────────────────────────────
def build_roc_data(preds_df):
    """Returns list of (fpr,tpr,auc) per fold + (mean_fpr, mean_tpr, std_tpr, mean_auc)."""
    fold_data = []
    mean_fpr  = np.linspace(0, 1, 300)
    tprs      = []

    for fold in sorted(preds_df["fold"].unique()):
        fd = preds_df[preds_df["fold"]==fold]
        y  = fd["label"].values
        p  = fd["prob_repo"].values
        if len(set(y)) < 2: continue
        fpr,tpr,_ = roc_curve(y,p)
        auc = roc_auc_score(y,p)
        fold_data.append((fpr,tpr,auc))
        tprs.append(np.interp(mean_fpr,fpr,tpr))

    mean_tpr = np.mean(tprs,axis=0) if tprs else None
    std_tpr  = np.std(tprs,axis=0)  if tprs else None
    mean_auc = float(np.trapz(mean_tpr,mean_fpr)) if mean_tpr is not None else None
    return fold_data, mean_fpr, mean_tpr, std_tpr, mean_auc


def build_pr_data(preds_df):
    """Returns list of (rec,prec,ap) per fold + mean."""
    fold_data = []
    mean_rec  = np.linspace(0,1,300)
    precs     = []

    for fold in sorted(preds_df["fold"].unique()):
        fd = preds_df[preds_df["fold"]==fold]
        y  = fd["label"].values
        p  = fd["prob_repo"].values
        if len(set(y)) < 2: continue
        prec,rec,_ = precision_recall_curve(y,p)
        ap = average_precision_score(y,p)
        fold_data.append((rec,prec,ap))
        # Interpolate precision on common recall grid (flip for interp)
        precs.append(np.interp(mean_rec, rec[::-1], prec[::-1]))

    mean_prec = np.mean(precs,axis=0) if precs else None
    std_prec  = np.std(precs,axis=0)  if precs else None
    mean_ap   = float(np.mean([d[2] for d in fold_data])) if fold_data else None
    return fold_data, mean_rec, mean_prec, std_prec, mean_ap


# ── Figure 1: ROC — 1×3 subplots ──────────────────────────────────────────────
def fig_roc_3runs(run_data, out_path):
    runs = [r for r in ["all","male","female"] if r in run_data and run_data[r][0] is not None]

    # Layout: 2 on top row, 1 centred on bottom row
    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs  = fig.add_gridspec(2, 4)
    if len(runs) == 3:
        axes = [
            fig.add_subplot(gs[0, 0:2]),   # top-left
            fig.add_subplot(gs[0, 2:4]),   # top-right
            fig.add_subplot(gs[1, 1:3]),   # bottom-centre
        ]
    else:
        axes = [fig.add_subplot(gs[0, i*2:(i+1)*2]) for i in range(len(runs))]

    panel_labels = ["A","B","C"]
    for panel_idx, (ax, run) in enumerate(zip(axes, runs)):
        preds = run_data[run][0]
        fold_data, mfpr, mtpr, stpr, mauc = build_roc_data(preds)

        for i,(fpr,tpr,auc) in enumerate(fold_data):
            ax.plot(fpr,tpr,lw=0.9,alpha=0.5,color=FOLD_COLORS[i],
                    label=f"Fold {i} ({auc:.3f})")

        if mtpr is not None:
            ax.plot(mfpr,mtpr,lw=2.2,color=MEAN_COLOR,
                    label=f"Mean AUC={mauc:.3f}±{np.std([d[2] for d in fold_data]):.3f}")
            ax.fill_between(mfpr,
                            np.clip(mtpr-stpr,0,1),
                            np.clip(mtpr+stpr,0,1),
                            alpha=FILL_ALPHA, color=MEAN_COLOR)

        ax.plot([0,1],[0,1],"--",color="#aaaaaa",lw=1,zorder=0)
        ax.set_xlim([0,1]); ax.set_ylim([0,1.01])
        ax.set_xlabel("False Positive Rate")
        ax.set_title(f"{panel_labels[panel_idx]}   {RUN_LABELS.get(run,run)}",
                     loc="left")
        ax.grid(True,alpha=0.4)
        ax.legend(loc="lower right", fontsize=8)

    for ax in axes:
        ax.set_ylabel("True Positive Rate")

    fig.suptitle("ROC Curves — Run 1 Model (5-Fold Evaluation)",
                 fontsize=14, fontweight="bold")

    for ext in [".png",".pdf"]:
        fig.savefig(str(out_path)+ext)
    plt.close(fig)
    print(f"  Saved: {out_path}.png/.pdf")


# ── Figure 2: PR — 1×3 subplots ───────────────────────────────────────────────
def fig_pr_3runs(run_data, out_path):
    runs = [r for r in ["all","male","female"] if r in run_data and run_data[r][0] is not None]

    # Layout: 2 on top row, 1 centred on bottom row
    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs  = fig.add_gridspec(2, 4)
    if len(runs) == 3:
        axes = [
            fig.add_subplot(gs[0, 0:2]),
            fig.add_subplot(gs[0, 2:4]),
            fig.add_subplot(gs[1, 1:3]),
        ]
    else:
        axes = [fig.add_subplot(gs[0, i*2:(i+1)*2]) for i in range(len(runs))]

    panel_labels = ["A","B","C"]
    for panel_idx, (ax, run) in enumerate(zip(axes, runs)):
        preds = run_data[run][0]
        fold_data, mrec, mprec, sprec, map_ = build_pr_data(preds)
        baseline = float(preds["label"].mean())

        for i,(rec,prec,ap) in enumerate(fold_data):
            ax.plot(rec,prec,lw=0.9,alpha=0.5,color=FOLD_COLORS[i],
                    label=f"Fold {i} ({ap:.3f})")

        if mprec is not None:
            ap_vals = [d[2] for d in fold_data]
            ax.plot(mrec,mprec,lw=2.2,color=MEAN_COLOR,
                    label=f"Mean AP={map_:.3f}±{np.std(ap_vals):.3f}")
            ax.fill_between(mrec,
                            np.clip(mprec-sprec,0,1),
                            np.clip(mprec+sprec,0,1),
                            alpha=FILL_ALPHA, color=MEAN_COLOR)

        ax.axhline(baseline,color="#aaaaaa",ls="--",lw=1,
                   label=f"Baseline={baseline:.2f}",zorder=0)
        ax.set_xlim([0,1]); ax.set_ylim([0,1.01])
        ax.set_xlabel("Recall")
        ax.set_title(f"{panel_labels[panel_idx]}   {RUN_LABELS.get(run,run)}",
                     loc="left")
        ax.grid(True,alpha=0.4)
        ax.legend(loc="upper right", fontsize=8)

    for ax in axes:
        ax.set_ylabel("Precision")
    fig.suptitle("Precision-Recall Curves — Run 1 Model (5-Fold Evaluation)",
                 fontsize=14, fontweight="bold")

    for ext in [".png",".pdf"]:
        fig.savefig(str(out_path)+ext)
    plt.close(fig)
    print(f"  Saved: {out_path}.png/.pdf")



# ── Figure 4: Metrics tables ───────────────────────────────────────────────────
# ── Figure 3: Noise-injection robustness ──────────────────────────────────────
NOISE_LABELS = {
    "gaussian":   "Gaussian noise (sigma)",
    "blur":       "Gaussian blur (kernel)",
    "brightness": "Brightness factor",
}


def fig_noise_injection(noise_csv: Path, out_path: Path):
    """
    AUROC against perturbation severity, one panel per noise type, one line per
    cohort. Reads the CSV written by noise_robustness.py, which must contain the
    columns: run, noise_type, severity, auroc.
    """
    df = pd.read_csv(noise_csv)
    required = {"run", "noise_type", "severity", "auroc"}
    missing = required - set(df.columns)
    if missing:
        print(f"  Skipping noise figure — CSV missing columns: {sorted(missing)}")
        return

    noise_types = [n for n in ["gaussian", "blur", "brightness"]
                   if n in set(df["noise_type"])]
    if not noise_types:
        print("  Skipping noise figure — no perturbation rows found")
        return

    runs = [r for r in ["all", "male", "female"] if r in set(df["run"])]
    runs += [r for r in sorted(set(df["run"])) if r not in runs]

    fig, axes = plt.subplots(1, len(noise_types),
                             figsize=(4.2 * len(noise_types), 3.8))
    if len(noise_types) == 1:
        axes = [axes]

    for ax, ntype in zip(axes, noise_types):
        for run in runs:
            sub = (df[(df["run"] == run) & (df["noise_type"] == ntype)]
                   .sort_values("severity"))
            if sub.empty:
                continue

            base = df[(df["run"] == run) & (df["noise_type"] == "clean")]["auroc"]
            xs = list(sub["severity"])
            ys = list(sub["auroc"])
            if len(base):
                xs = [0] + xs
                ys = [float(base.iloc[0])] + ys

            ax.plot(xs, ys, marker="o", lw=1.8, ms=5,
                    color=RUN_COLORS.get(run), label=RUN_LABELS.get(run, run))

        ax.set_title(NOISE_LABELS.get(ntype, ntype))
        ax.set_xlabel("Severity")
        ax.set_ylim(0.4, 1.0)
        ax.axhline(0.5, color="#999999", ls="--", lw=0.8, zorder=0)
        ax.grid(True, alpha=0.4)

    axes[0].set_ylabel("AUROC")
    axes[-1].legend(loc="lower left", fontsize=8)
    fig.suptitle("Robustness to image degradation", y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_path}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out_path.name}.pdf/.png")


def fig_metrics_table(metrics_df, run, out_path):
    """Clean publication-style metrics table for one run."""
    cols_display = {
        "fold":          "Fold",
        "tile_auroc_t05":"AUROC",
        "tile_ap_t05" if "tile_ap_t05" in metrics_df.columns else "tile_auroc_t05": "AP",
        "tile_f1_t05":   "F1",
        "tile_sens_t05": "Sensitivity",
        "tile_spec_t05": "Specificity",
        "tile_ppv_t05":  "PPV",
        "tile_npv_t05":  "NPV",
        "slide_auroc":   "Slide AUROC",
    }
    cols_display = {k:v for k,v in cols_display.items() if k in metrics_df.columns}

    df = metrics_df[list(cols_display.keys())].copy()
    df = df.rename(columns=cols_display)
    df["Fold"] = [f"Fold {int(f)}" for f in df["Fold"]]

    # Add mean±std row
    num_cols = [c for c in df.columns if c != "Fold"]
    mean_row = {"Fold": "Mean±SD"}
    for c in num_cols:
        vals = pd.to_numeric(df[c], errors="coerce").dropna()
        mean_row[c] = f"{vals.mean():.3f}±{vals.std():.3f}"
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    # Round numerics
    for c in num_cols:
        df[c] = df[c].apply(lambda x:
            f"{float(x):.4f}" if isinstance(x,float) else x)

    fig_h = max(2.5, 0.5*(len(df)+2))
    fig, ax = plt.subplots(figsize=(max(10,len(df.columns)*1.5), fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)

    # Style header
    for j in range(len(df.columns)):
        cell = tbl[(0,j)]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Style mean row (last row)
    for j in range(len(df.columns)):
        cell = tbl[(len(df),j)]
        cell.set_facecolor("#ecf0f1")
        cell.set_text_props(fontweight="bold")

    # Alternating row colors
    for i in range(1, len(df)):
        for j in range(len(df.columns)):
            tbl[(i,j)].set_facecolor(
                "#f8f9fa" if i%2==0 else "white")

    ax.set_title(f"{RUN_LABELS.get(run,run)} — Metrics Summary (t=0.5)",
                 fontsize=12, fontweight="bold", pad=15)
    fig.tight_layout()

    for ext in [".png",".pdf"]:
        fig.savefig(str(out_path)+ext, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}.png/.pdf")


# ── Figure 5: Combined summary table ──────────────────────────────────────────
def fig_summary_table(run_data, out_path):
    rows = []
    for run in ["all","male","female"]:
        if run not in run_data or run_data[run][1] is None: continue
        mdf = run_data[run][1]
        row = {"Run": RUN_LABELS.get(run,run)}
        for col in ["tile_auroc_t05","tile_f1_t05","tile_sens_t05",
                    "tile_spec_t05","tile_ppv_t05","tile_npv_t05","slide_auroc"]:
            if col not in mdf.columns: continue
            vals = pd.to_numeric(mdf[col],errors="coerce").dropna().values
            label = col.replace("tile_","").replace("_t05","").replace("_"," ").title()
            row[label] = f"{np.mean(vals):.3f}±{np.std(vals):.3f}"
        rows.append(row)

    df = pd.DataFrame(rows)
    fig_h = max(2.5, 0.5*(len(df)+2))
    fig, ax = plt.subplots(figsize=(max(12,len(df.columns)*1.8), fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1,2.0)
    for j in range(len(df.columns)):
        c = tbl[(0,j)]
        c.set_facecolor("#2c3e50")
        c.set_text_props(color="white",fontweight="bold")
    for i in range(1,len(df)+1):
        for j in range(len(df.columns)):
            tbl[(i,j)].set_facecolor(
                RUN_COLORS.get(list(run_data.keys())[i-1],"#ffffff")+"22"
                if i<=len(run_data) else "white")
    ax.set_title("Cross-Cohort Metrics Summary — Run 1 Model (Mean±SD across 5 folds)",
                 fontsize=12,fontweight="bold",pad=15)
    fig.tight_layout()
    for ext in [".png",".pdf"]:
        fig.savefig(str(out_path)+ext,bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}.png/.pdf")


# ── Figure 6: Full metrics table (tile + slide, per fold + pooled) ────────────
def compute_full_metrics(y, p, level="tile"):
    """Compute all metrics at Youden threshold."""
    from sklearn.metrics import (accuracy_score, f1_score,
                                 confusion_matrix as cm_fn2)
    if len(set(y)) < 2:
        return None
    fpr, tpr, thr = roc_curve(y, p)
    opt = float(thr[np.argmax(tpr - fpr)])
    auroc = roc_auc_score(y, p)
    yp = (p >= opt).astype(int)
    tn, fp, fn, tp = cm_fn2(y, yp, labels=[0,1]).ravel()
    return {
        "auroc":       round(auroc, 4),
        "accuracy":    round(accuracy_score(y, yp), 4),
        "f1":          round(f1_score(y, yp, zero_division=0), 4),
        "sensitivity": round(tp/(tp+fn) if tp+fn else 0., 4),
        "specificity": round(tn/(tn+fp) if tn+fp else 0., 4),
        "ppv":         round(tp/(tp+fp) if tp+fp else 0., 4),
        "npv":         round(tn/(tn+fn) if tn+fn else 0., 4),
        "youden_thr":  round(opt, 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "n":  len(y),
    }


def fig_full_metrics_table(run_data, out_path):
    """
    Publication table: rows = Fold 0..4 + Pooled + Mean±SD
                       cols = metric
                       sections = tile level / slide level
                       panels = one per run
    """
    metric_cols  = ["youden_thr","auroc","accuracy","f1",
                    "sensitivity","specificity","ppv","npv"]
    metric_labels = {
        "youden_thr":  "Youden Thr.",
        "auroc":       "AUROC",
        "accuracy":    "Accuracy",
        "f1":          "F1",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
        "ppv":         "PPV",
        "npv":         "NPV",
    }
    runs = [r for r in ["all","male","female"]
            if r in run_data and run_data[r][0] is not None]

    # One figure per level (tile / slide)
    for level in ["tile", "slide"]:
        all_run_tables = {}

        for run in runs:
            preds = run_data[run][0]
            rows  = []

            folds = sorted(preds["fold"].unique())
            fold_aurocs = []  # for mean±SD

            for fold in folds:
                fd = preds[preds["fold"]==fold]
                y  = fd["label"].values
                p  = fd["prob_repo"].values

                if level == "tile":
                    m = compute_full_metrics(y, p)
                else:
                    # Slide level aggregation
                    sdf = fd.groupby("slide_id").agg(
                        mean_prob=("prob_repo","mean"),
                        label=("label","first")).reset_index()
                    m = compute_full_metrics(
                        sdf["label"].values, sdf["mean_prob"].values)

                if m is None:
                    row = {"row": f"Fold {fold}"}
                    row.update({c: "—" for c in metric_cols})
                else:
                    fold_aurocs.append(m["auroc"])
                    row = {"row": f"Fold {fold}"}
                    row.update({c: f"{m[c]:.4f}" for c in metric_cols})
                rows.append(row)

            # Pooled (all folds combined)
            y_all = preds["label"].values
            p_all = preds["prob_repo"].values
            if level == "tile":
                m_pool = compute_full_metrics(y_all, p_all)
            else:
                sdf_all = preds.groupby("slide_id").agg(
                    mean_prob=("prob_repo","mean"),
                    label=("label","first")).reset_index()
                m_pool = compute_full_metrics(
                    sdf_all["label"].values, sdf_all["mean_prob"].values)

            pool_row = {"row": "Pooled"}
            pool_row.update({c: f"{m_pool[c]:.4f}" for c in metric_cols}
                            if m_pool else {c:"—" for c in metric_cols})
            rows.append(pool_row)

            # Mean±SD row
            mean_row = {"row": "Mean±SD"}
            for c in metric_cols:
                vals = []
                for r in rows[:-1]:  # exclude pooled
                    try: vals.append(float(r[c]))
                    except: pass
                if vals:
                    mean_row[c] = f"{np.mean(vals):.4f}±{np.std(vals):.4f}"
                else:
                    mean_row[c] = "—"
            rows.append(mean_row)

            all_run_tables[run] = rows

        # Build figure — one panel per run stacked vertically
        n_rows_per_run = len(list(all_run_tables.values())[0])  # folds+pooled+mean
        n_cols = len(metric_cols) + 1  # +1 for row label

        fig_h = max(4, 0.55 * (n_rows_per_run * len(runs) + len(runs)*2))
        fig, axes = plt.subplots(len(runs), 1,
                                  figsize=(max(14, n_cols*1.6), fig_h),
                                  constrained_layout=True)
        if len(runs) == 1: axes = [axes]

        header = [""] + [metric_labels[c] for c in metric_cols]

        for ax, run in zip(axes, runs):
            rows = all_run_tables[run]
            cell_data = [[r["row"]] + [r[c] for c in metric_cols] for r in rows]

            ax.axis("off")
            tbl = ax.table(
                cellText=cell_data,
                colLabels=header,
                cellLoc="center",
                loc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.scale(1, 1.7)

            # Header row style
            for j in range(n_cols):
                c = tbl[(0, j)]
                c.set_facecolor("#2c3e50")
                c.set_text_props(color="white", fontweight="bold")

            # Alternating rows
            for i in range(1, n_rows_per_run + 1):
                row_label = cell_data[i-1][0]
                for j in range(n_cols):
                    cell = tbl[(i, j)]
                    if row_label == "Pooled":
                        cell.set_facecolor("#d5e8f7")
                        cell.set_text_props(fontweight="bold")
                    elif row_label == "Mean±SD":
                        cell.set_facecolor("#ecf0f1")
                        cell.set_text_props(fontweight="bold", fontstyle="italic")
                    else:
                        cell.set_facecolor("#f8f9fa" if (i%2==0) else "white")

            ax.set_title(RUN_LABELS.get(run, run),
                         fontsize=11, fontweight="bold",
                         loc="left", pad=8)

        level_label = "Tile" if level=="tile" else "Slide"
        fig.suptitle(
            f"Metrics Summary — {level_label} Level | Youden Optimal Threshold"
            " (AUROC consistent with ROC curves)",
            fontsize=13, fontweight="bold")

        for ext in [".png", ".pdf"]:
            fig.savefig(str(out_path) + f"_{level}" + ext)
        plt.close(fig)
        print(f"  Saved: {Path(out_path).name}_{level}.png/.pdf")

    # Also save raw CSV
    csv_rows = []
    for level in ["tile","slide"]:
        for run in runs:
            preds = run_data[run][0]
            for fold in sorted(preds["fold"].unique()):
                fd = preds[preds["fold"]==fold]
                y = fd["label"].values
                p = fd["prob_repo"].values
                if level=="tile":
                    m = compute_full_metrics(y,p)
                else:
                    sdf = fd.groupby("slide_id").agg(
                        mean_prob=("prob_repo","mean"),
                        label=("label","first")).reset_index()
                    m = compute_full_metrics(sdf["label"].values,
                                            sdf["mean_prob"].values)
                if m:
                    csv_rows.append({"level":level,"run":run,
                                     "fold":fold,**m})
            # Pooled
            y_all = preds["label"].values
            p_all = preds["prob_repo"].values
            if level=="tile":
                m = compute_full_metrics(y_all,p_all)
            else:
                sdf = preds.groupby("slide_id").agg(
                    mean_prob=("prob_repo","mean"),
                    label=("label","first")).reset_index()
                m = compute_full_metrics(sdf["label"].values,
                                         sdf["mean_prob"].values)
            if m:
                csv_rows.append({"level":level,"run":run,"fold":"pooled",**m})

    pd.DataFrame(csv_rows).to_csv(str(out_path)+"_raw.csv",index=False)
    print(f"  Saved: {Path(out_path).name}_raw.csv")


# ── Figure 7: Confusion matrices — all three runs ─────────────────────────────
def fig_confusion_matrices(run_data, out_path, test_pred_paths=None):
    """
    2x2 layout: first cohort top-left, second top-right, third bottom-centre.
    If test_pred_paths dict provided, uses those CSVs (test set only).
    Otherwise falls back to pooled all_fold_predictions.
    """
    from sklearn.metrics import confusion_matrix as cm_fn

    runs = [r for r in ["all","male","female"]
            if r in run_data and run_data[r][0] is not None]

    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs  = fig.add_gridspec(2, 4)
    if len(runs) == 3:
        axes = [
            fig.add_subplot(gs[0, 0:2]),
            fig.add_subplot(gs[0, 2:4]),
            fig.add_subplot(gs[1, 1:3]),
        ]
    else:
        axes = [fig.add_subplot(gs[0, i*2:(i+1)*2]) for i in range(len(runs))]

    class_names = ["ctrl", "rEPO"]

    for ax, run in zip(axes, runs):
        # Use test-set predictions if provided, else pooled folds
        if test_pred_paths and run in test_pred_paths:
            preds = pd.read_csv(test_pred_paths[run])
            data_label = "Test Set Only"
        else:
            preds = run_data[run][0]
            data_label = "All Folds (Pooled)"
        y = preds["label"].values
        p = preds["prob_repo"].values

        # Youden threshold on pooled
        fpr, tpr, thr = roc_curve(y, p)
        opt = float(thr[np.argmax(tpr - fpr)])
        auroc = roc_auc_score(y, p)
        yp = (p >= opt).astype(int)

        cm = cm_fn(y, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp/(tp+fn) if tp+fn else 0.
        spec = tn/(tn+fp) if tn+fp else 0.

        im = ax.imshow(cm, cmap="Blues", vmin=0)

        # Cell annotations
        thresh = cm.max() / 2
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        fontsize=22, fontweight="bold",
                        color="white" if cm[i,j] > thresh else "#333333")

        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(class_names, fontsize=12)
        ax.set_yticklabels(class_names, fontsize=12)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title(
            f"{RUN_LABELS.get(run, run)}\n"
            f"AUROC={auroc:.4f} | Thr={opt:.3f}\n"
            f"Sens={sens:.4f} | Spec={spec:.4f}\n"
            f"n={len(y)} ({data_label})",
            fontsize=10, fontweight="bold"
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Confusion Matrices — Pooled (All Folds) | Youden Optimal Threshold",
        fontsize=14, fontweight="bold"
    )

    for ext in [".png", ".pdf"]:
        fig.savefig(str(out_path) + ext)
    plt.close(fig)
    print(f"  Saved: {Path(out_path).name}.png/.pdf")


# ── Figure: Combined ROC + PR — 3 rows × 2 cols ───────────────────────────────
def fig_roc_pr_combined(run_data, out_path):
    """
    3 rows × 2 columns:
      col 0 = ROC curve  |  col 1 = PR curve
      row 0 = Run 1      |  row 1 = Run 2  |  row 2 = Run 3
    Panel labels A–F down the left then right column.
    """
    runs = [r for r in ["all","male","female"]
            if r in run_data and run_data[r][0] is not None]
    n    = len(runs)

    fig, axes = plt.subplots(n, 2,
                              figsize=(12, 5.5 * n),
                              constrained_layout=True)
    if n == 1:
        axes = axes[None, :]   # keep 2-D indexing

    # Panel labels A–F: left column first (A,B,C), then right (D,E,F)
    panel_labels = [chr(65 + i) for i in range(n * 2)]  # A B C D E F
    # Assign: axes[0,0]=A, axes[1,0]=B, axes[2,0]=C,
    #         axes[0,1]=D, axes[1,1]=E, axes[2,1]=F
    label_map = {}
    for row in range(n):
        label_map[(row, 0)] = panel_labels[row]
        label_map[(row, 1)] = panel_labels[row + n]

    mean_fpr = np.linspace(0, 1, 300)
    mean_rec = np.linspace(0, 1, 300)

    for row, run in enumerate(runs):
        preds    = run_data[run][0]
        short_lbl = {"all": "All participants",
                  "male": "Male participants",
                  "female": "Female participants"}.get(run, run)

        # ── ROC ──────────────────────────────────────────────────────────────
        ax_roc = axes[row, 0]
        tprs, aucs = [], []
        for fold in sorted(preds["fold"].unique()):
            fd = preds[preds["fold"] == fold]
            y, p = fd["label"].values, fd["prob_repo"].values
            if len(set(y)) < 2: continue
            fpr, tpr, _ = roc_curve(y, p)
            auc = roc_auc_score(y, p)
            ax_roc.plot(fpr, tpr, lw=1.8, alpha=0.6,
                        color=FOLD_COLORS[fold % len(FOLD_COLORS)],
                        label=f"Fold {fold} (AUC={auc:.3f})")
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            aucs.append(auc)

        if tprs:
            mt = np.mean(tprs, axis=0); st = np.std(tprs, axis=0)
            ax_roc.plot(mean_fpr, mt, lw=2.8, color="#000000",
                        label=f"Mean AUC = {np.mean(aucs):.3f}±{np.std(aucs):.3f}")
            ax_roc.fill_between(mean_fpr,
                                np.clip(mt-st,0,1), np.clip(mt+st,0,1),
                                alpha=FILL_ALPHA, color="#000000")

        ax_roc.plot([0,1],[0,1],"--",color="#aaa",lw=0.8)
        ax_roc.set_xlim([0,1]); ax_roc.set_ylim([0,1.01])
        ax_roc.set_xlabel("False Positive Rate", fontsize=11)
        ax_roc.set_ylabel("True Positive Rate", fontsize=11)
        ax_roc.set_title(f"{label_map[(row,0)]} {short_lbl} — ROC",
                          fontsize=12, fontweight="bold", loc="center")
        ax_roc.legend(loc="lower right", fontsize=8)
        ax_roc.grid(True, alpha=0.4)

        # ── PR ───────────────────────────────────────────────────────────────
        ax_pr  = axes[row, 1]
        precs, aps = [], []
        baseline = float(preds["label"].mean())

        for fold in sorted(preds["fold"].unique()):
            fd = preds[preds["fold"] == fold]
            y, p = fd["label"].values, fd["prob_repo"].values
            if len(set(y)) < 2: continue
            prec, rec, _ = precision_recall_curve(y, p)
            ap = average_precision_score(y, p)
            ax_pr.plot(rec, prec, lw=1.8, alpha=0.6,
                       color=FOLD_COLORS[fold % len(FOLD_COLORS)],
                       label=f"Fold {fold} (AP={ap:.3f})")
            precs.append(np.interp(mean_rec, rec[::-1], prec[::-1]))
            aps.append(ap)

        if precs:
            mp = np.mean(precs, axis=0); sp = np.std(precs, axis=0)
            ax_pr.plot(mean_rec, mp, lw=2.8, color="#000000",
                       label=f"Mean AP = {np.mean(aps):.3f}±{np.std(aps):.3f}")
            ax_pr.fill_between(mean_rec,
                               np.clip(mp-sp,0,1), np.clip(mp+sp,0,1),
                               alpha=FILL_ALPHA, color="#000000")

        ax_pr.axhline(baseline, color="#aaa", ls="--", lw=0.8,
                      label=f"Baseline = {baseline:.3f}")
        ax_pr.set_xlim([0,1]); ax_pr.set_ylim([0,1.01])
        ax_pr.set_xlabel("Recall", fontsize=11)
        ax_pr.set_ylabel("Precision", fontsize=11)
        ax_pr.set_title(f"{label_map[(row,1)]} {short_lbl} — PR",
                         fontsize=12, fontweight="bold", loc="center")
        ax_pr.legend(loc="lower left", fontsize=8)
        ax_pr.grid(True, alpha=0.4)

    for ext in [".png", ".pdf"]:
        fig.savefig(str(out_path) + ext, dpi=300)
    plt.close(fig)
    print(f"  Saved: {Path(out_path).name}.png/.pdf")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True,
                    help="Output dir from the evaluation step")
    ap.add_argument("--out_dir",     required=True)
    ap.add_argument("--noise_csv",   default=None,
                    help="noise_results_all.csv produced by noise_robustness.py")
    ap.add_argument("--test_preds_dir", default=None,
                    help="Directory holding per-cohort test tile_predictions.csv")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"WADA Publication Figures")
    print(f"Results : {results_dir}")
    print(f"Output  : {out_dir}")
    print(f"{'='*60}")

    # Load all run data
    run_data = {}
    for run in ["all","male","female"]:
        preds, metrics = load_run_data(results_dir, run)
        if preds is not None:
            run_data[run] = (preds, metrics)
            print(f"  Loaded {run}: {len(preds)} tiles across "
                  f"{preds['fold'].nunique()} folds")
        else:
            print(f"  {run}: not found — skipping")

    if not run_data:
        print("ERROR: no run data found"); return

    # ── Generate figures ───────────────────────────────────────────────────────
    print(f"\n── Generating figures")

    # print("\n1. ROC curves (separate)")
    # fig_roc_3runs(run_data, out_dir/"roc_3runs")

    # print("\n2. PR curves (separate)")
    # fig_pr_3runs(run_data, out_dir/"pr_3runs")

    print("\n1+2. Combined ROC + PR (3 rows × 2 cols)")
    fig_roc_pr_combined(run_data, out_dir/"roc_pr_combined")

    print("\n3. Noise-injection robustness")
    if args.noise_csv and Path(args.noise_csv).exists():
        fig_noise_injection(Path(args.noise_csv), out_dir/"noise_injection")
    else:
        print("  Skipped — pass --noise_csv with the output of "
              "noise_robustness.py")

    # print("\n4. Per-run metrics tables")
    # for run,(preds,metrics) in run_data.items():
    #     if metrics is not None:
    #         fig_metrics_table(metrics, run, out_dir/f"metrics_table_{run}")

    # print("\n5. Combined summary table")
    # fig_summary_table(run_data, out_dir/"summary_table")

    # print("\n6. Full per-fold metrics tables (tile + slide level)")
    # fig_full_metrics_table(run_data, out_dir/"full_metrics_table")

    print("\n7. Confusion matrices (test set only)")
    test_preds = {}
    if args.test_preds_dir:
        base = Path(args.test_preds_dir)
        for run in run_data:
            for cand in [base / run / "tile_predictions.csv",
                         base / f"tile_predictions_{run}.csv"]:
                if cand.exists():
                    test_preds[run] = cand
                    break
            else:
                print(f"  WARNING: test predictions not found for {run} under {base}")
    fig_confusion_matrices(run_data, out_dir/"confusion_matrices_3runs",
                           test_pred_paths=test_preds if test_preds else None)

    # Save combined summary CSV
    rows = []
    for run,(preds,metrics) in run_data.items():
        if metrics is None: continue
        for col in metrics.select_dtypes("number").columns:
            vals = metrics[col].dropna().values
            rows.append({"run":run,"metric":col,
                         "mean":round(float(np.mean(vals)),4),
                         "std": round(float(np.std(vals)),4)})
    pd.DataFrame(rows).to_csv(out_dir/"all_metrics_summary.csv",index=False)
    print(f"\n  Saved: all_metrics_summary.csv")

    print(f"\n{'='*60}")
    print(f"All figures saved to: {out_dir}")
    print(f"  roc_pr_combined.pdf/.png")
    print(f"  noise_injection.pdf/.png")
    print(f"  confusion_matrices_3runs.pdf/.png")
    print(f"  all_metrics_summary.csv")


if __name__ == "__main__":
    main()
