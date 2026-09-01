#!/usr/bin/env python3
"""
Regenerate noise-robustness figures from a previously computed results CSV.

Usage:
    python src/visualization/plot_noise_heatmap.py \
        --csv     outputs/noise/noise_results_all.csv \
        --out_dir outputs/figures
"""


import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── Blue publication style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#333333",
    "axes.linewidth":    1.2,
    "axes.labelcolor":   "black",
    "axes.labelsize":    12,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.titlecolor":   "black",
    "xtick.color":       "black",
    "ytick.color":       "black",
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "xtick.labelcolor":  "black",
    "ytick.labelcolor":  "black",
    "grid.color":        "#dddddd",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "legend.fontsize":   9,
    "legend.framealpha": 0.95,
    "legend.edgecolor":  "#cccccc",
    "legend.labelcolor": "black",
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica","Arial","DejaVu Sans"],
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
    "pdf.fonttype":      42,
})

# Blue palette — distinct shades per run
RUN_LABELS  = {
    "all": "All participants",
    "male": "Male participants",
    "female": "Female participants",
}
RUN_COLORS  = {
    "all": "#1a5fa8",   # deep blue
    "male": "#3a9bd5",   # medium blue
    "female": "#76c0e8",   # light blue
}
RUN_MARKERS = {"all": "o", "male": "s", "female": "^"}

NOISE_META = {
    "gaussian":   {
        "label":  "Gaussian Noise",
        "xlabel": "Noise level (σ)",
        "x_vals": [0, 10, 25, 50],
        "x_ticks":["Clean","σ=10","σ=25","σ=50"],
    },
    "blur":       {
        "label":  "Gaussian Blur",
        "xlabel": "Kernel size (px)",
        "x_vals": [0, 3, 7, 15],
        "x_ticks":["Clean","k=3","k=7","k=15"],
    },
    "brightness": {
        "label":  "Brightness Shift",
        "xlabel": "Scale factor",
        "x_vals": [1.0, 0.6, 0.8, 1.4],
        "x_ticks":["Clean","×0.6","×0.8","×1.4"],
    },
}


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Loaded: {len(df)} rows | runs: {df['run'].unique().tolist()}")
    print(f"Noise types: {df['noise_type'].unique().tolist()}")
    return df


# ── Figure 1: Line plots — AUROC vs severity ──────────────────────────────────
def fig_lineplot(df, out_path):
    noise_types = ["gaussian", "blur", "brightness"]
    runs = sorted(df["run"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)

    for ax, nt in zip(axes, noise_types):
        meta = NOISE_META[nt]
        clean_df = df[df["noise_type"] == "clean"]
        noisy_df = df[df["noise_type"] == nt]

        for run in runs:
            # Baseline AUROC
            bl = clean_df[clean_df["run"]==run]["auroc"].values
            if not len(bl): continue
            baseline = float(bl[0])

            # Noisy AUROCs in order
            sub = noisy_df[noisy_df["run"]==run].copy()
            if sub.empty: continue

            y_vals = [baseline] + list(sub["auroc"].values)
            x_pos  = list(range(len(y_vals)))

            ax.plot(x_pos, y_vals,
                    marker=RUN_MARKERS.get(run,"o"),
                    color=RUN_COLORS.get(run,"#1a5fa8"),
                    lw=2.2, ms=8, mec="white", mew=1.2,
                    label=RUN_LABELS.get(run, run),
                    zorder=3)

            # Shade area under baseline
            ax.fill_between(x_pos, y_vals, [baseline]*len(y_vals),
                            alpha=0.06,
                            color=RUN_COLORS.get(run,"#1a5fa8"))

        # Reference lines
        ax.axhline(0.5, color="#a0b8d4", ls=":", lw=1.2, zorder=1)
        ax.axhline(0.7, color="#c5d8ee", ls="--", lw=1.0, zorder=1,
                   label="AUROC=0.70")

        ax.set_xticks(range(4))
        ax.set_xticklabels(meta["x_ticks"], rotation=15,
                           ha="right", fontsize=9)
        ax.set_xlabel(meta["xlabel"], color="#000000")
        ax.set_ylabel("AUROC" if ax is axes[0] else "",
                      color="#000000")
        ax.set_title(meta["label"])
        ax.set_ylim([max(0, df["auroc"].min()-0.08), 1.02])
        ax.grid(True, alpha=0.5, zorder=0)
        ax.set_facecolor("#ffffff")

        if ax is axes[0]:
            ax.legend(loc="lower left", framealpha=0.9)

    fig.suptitle("Noise Robustness — AUROC vs Perturbation Severity",
                 fontsize=14, fontweight="bold", color="#000000")

    for ext in [".png", ".pdf"]:
        fig.savefig(str(out_path) + ext, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {Path(out_path).name}.png/.pdf")


# ── Figure 2: Heatmap — AUROC drop ────────────────────────────────────────────
def fig_heatmap(df, out_path):
    runs  = sorted(df["run"].unique())
    noisy = df[df["noise_type"] != "clean"].copy()

    order = [
        ("gaussian",   "Low (σ=10)"),
        ("gaussian",   "Med (σ=25)"),
        ("gaussian",   "High (σ=50)"),
        ("blur",       "Low (k=3)"),
        ("blur",       "Med (k=7)"),
        ("blur",       "High (k=15)"),
        ("brightness", "Dark (0.6)"),
        ("brightness", "Dim (0.8)"),
        ("brightness", "Bright (1.4)"),
    ]

    sev_short = {
        "Low (σ=10)":   "Low (σ=10)",
        "Med (σ=25)":   "Med (σ=25)",
        "High (σ=50)":  "High (σ=50)",
        "Low (k=3)":    "Low (k=3)",
        "Med (k=7)":    "Med (k=7)",
        "High (k=15)":  "High (k=15)",
        "Dark (0.6)":   "Dark (×0.6)",
        "Dim (0.8)":    "Dim (×0.8)",
        "Bright (1.4)": "Bright (×1.4)",
    }
    row_labels = [sev_short.get(s, s) for _, s in order]

    col_labels = {
        "all": "All participants",
        "male": "Male participants",
        "female": "Female participants",
    }

    # Build matrix
    matrix = np.zeros((len(order), len(runs)))
    for i, (nt, sl) in enumerate(order):
        for j, run in enumerate(runs):
            val = noisy[(noisy["noise_type"]==nt) &
                        (noisy["severity_label"]==sl) &
                        (noisy["run"]==run)]["auroc_drop"].values
            matrix[i, j] = float(val[0]) if len(val) else 0.

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "blue_robust",
        ["#1a5fa8","#4a90c8","#c5d8ee","#f5c97a","#e07020"], N=256)

    fig, ax = plt.subplots(figsize=(max(7, len(runs)*3.0),
                                    max(6, len(order)*0.75)),
                            constrained_layout=True)
    fig.patch.set_facecolor("#f0f4f8")
    ax.set_facecolor("#f0f4f8")

    vmax = max(0.08, float(matrix.max()))
    im   = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Drop from clean baseline", color="#000000", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="#000000")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#000000")

    # X-axis on top with cohort names
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([col_labels.get(r, r) for r in runs],
                       fontsize=11, color="#000000", fontweight="bold")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    # Y-axis — short severity labels only
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(row_labels, fontsize=9, color="#000000")

    # Horizontal dividers between noise groups
    for div in [2.5, 5.5]:
        ax.axhline(div, color="#000000", lw=1.5, alpha=0.5, zorder=3)

    # Bracket annotations for noise groups
    group_info = [
        ("Gaussian Noise",   0, 2),
        ("Gaussian Blur",    3, 5),
        ("Brightness Shift", 6, 8),
    ]
    for grp_label, r_start, r_end in group_info:
        y_mid = (r_start + r_end) / 2
        y_top = r_start - 0.42
        y_bot = r_end   + 0.42
        x_brk = -1.1

        # Vertical bar
        ax.annotate("", xy=(x_brk, y_top), xytext=(x_brk, y_bot),
                    xycoords="data", textcoords="data",
                    arrowprops=dict(arrowstyle="-", color="#000000", lw=1.5),
                    annotation_clip=False)
        # Top serif
        ax.annotate("", xy=(x_brk, y_top), xytext=(x_brk+0.1, y_top),
                    xycoords="data", textcoords="data",
                    arrowprops=dict(arrowstyle="-", color="#000000", lw=1.5),
                    annotation_clip=False)
        # Bottom serif
        ax.annotate("", xy=(x_brk, y_bot), xytext=(x_brk+0.1, y_bot),
                    xycoords="data", textcoords="data",
                    arrowprops=dict(arrowstyle="-", color="#000000", lw=1.5),
                    annotation_clip=False)
        # Rotated group label
        ax.text(x_brk - 0.20, y_mid, grp_label,
                ha="right", va="center", fontsize=9.5,
                color="#000000", fontweight="bold", rotation=90,
                transform=ax.transData, clip_on=False)

    # Cell values
    for i in range(len(order)):
        for j in range(len(runs)):
            val = matrix[i, j]
            txt_color = "#ffffff" if val > vmax * 0.6 else "#000000"
            ax.text(j, i, f"{val:.3f}",
                    ha="center", va="center",
                    fontsize=10, color=txt_color, fontweight="bold")

    ax.tick_params(colors="#000000")
    for spine in ax.spines.values():
        spine.set_edgecolor("#000000")

    for ext in [".png", ".pdf"]:
        fig.savefig(str(out_path) + ext, dpi=300)
    plt.close(fig)
    print(f"  Saved: {Path(out_path).name}.png/.pdf")

# ── Figure 3: Summary bar — AUROC drop per run ────────────────────────────────
def fig_drop_summary(df, out_path):
    """
    Grouped bar chart: max AUROC drop per noise type × run.
    """
    runs = sorted(df["run"].unique())
    noise_types = ["gaussian","blur","brightness"]
    noise_labels = ["Gaussian Noise","Blur","Brightness"]
    noisy = df[df["noise_type"]!="clean"]

    x = np.arange(len(noise_types))
    width = 0.25
    offsets = np.linspace(-(len(runs)-1)*width/2,
                           (len(runs)-1)*width/2, len(runs))

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    fig.patch.set_facecolor("#f0f4f8")
    ax.set_facecolor("#ffffff")

    for run, offset in zip(runs, offsets):
        max_drops = []
        for nt in noise_types:
            sub = noisy[(noisy["run"]==run)&(noisy["noise_type"]==nt)]
            max_drops.append(float(sub["auroc_drop"].max()) if not sub.empty else 0.)

        bars = ax.bar(x + offset, max_drops, width*0.88,
                      label=RUN_LABELS.get(run,run),
                      color=RUN_COLORS.get(run,"#1a5fa8"),
                      edgecolor="white", linewidth=0.8,
                      alpha=0.88)

        for bar, val in zip(bars, max_drops):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.002,
                    f"{val:.3f}",
                    ha="center", va="bottom",
                    fontsize=8.5, color="#000000")

    # Reference thresholds
    ax.axhline(0.05, color="#3a9bd5", ls="--", lw=1.2,
               label="Robust threshold (0.05)", alpha=0.8)
    ax.axhline(0.10, color="#e07020", ls=":", lw=1.2,
               label="Fragile threshold (0.10)", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(noise_labels, fontsize=11, color="#000000")
    ax.set_ylabel("Max AUROC Drop (highest severity)", color="#000000")
    ax.set_ylim([0, max(0.15, float(noisy["auroc_drop"].max())+0.04)])
    ax.set_title("Maximum AUROC Drop by Noise Type",
                 fontsize=13, fontweight="bold", color="#000000")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    for ext in [".png",".pdf"]:
        fig.savefig(str(out_path)+ext)
    plt.close(fig)
    print(f"  Saved: {Path(out_path).name}.png/.pdf")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",     required=True,
                    help="noise_results_all.csv produced by noise_robustness.py")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"Noise Robustness Figures (blue scheme)")
    print(f"CSV    : {args.csv}")
    print(f"Output : {out_dir}")
    print(f"{'='*55}")

    df = load_data(args.csv)

    # print("\n── Line plots (AUROC vs severity)")
    # fig_lineplot(df, out_dir/"noise_lineplot")

    print("\n── Heatmap (AUROC drop)")
    fig_heatmap(df, out_dir/"noise_heatmap")

    # print("\n── Drop summary bar chart")
    # fig_drop_summary(df, out_dir/"noise_drop_summary")

    print(f"\nDone — all figures in: {out_dir}")


if __name__ == "__main__":
    main()
