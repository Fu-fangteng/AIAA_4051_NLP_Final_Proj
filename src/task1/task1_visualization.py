"""
Render Task 1 faithfulness curve from cached data.

This script is render-only: it reads
`aiaa4051/task1/faithfulness/faithfulness_data.pkl` and does not load GPT-2.
Run task1_faithfulness.py first to generate the pkl.
"""

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = ROOT / "aiaa4051" / "task1" / "faithfulness" / "faithfulness_data.pkl"
OUT_DIR = ROOT / "aiaa4051" / "task1" / "faithfulness"


def _mean_and_sem(curves):
    mean = curves.mean(axis=0)
    if len(curves) <= 1:
        return mean, np.zeros_like(mean)
    sem = curves.std(axis=0, ddof=1) / np.sqrt(len(curves))
    return mean, sem


def _plot_curve(ax, x_axis, curves, label, color, linestyle, zorder):
    mean, sem = _mean_and_sem(curves)
    ax.plot(
        x_axis,
        mean,
        label=label,
        color=color,
        linewidth=2.45,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.fill_between(
        x_axis,
        np.maximum(mean - sem, 0),
        mean + sem,
        color=color,
        alpha=0.11,
        linewidth=0,
        zorder=zorder - 1,
    )


def main():
    with open(DATA_PATH, "rb") as f:
        payload = pickle.load(f)

    attnlrp_arr = payload["attnlrp"]
    random_arr  = payload["random"]
    worst_arr   = payload["worst"]
    x           = payload["x"]
    auc         = payload["auc"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.edgecolor": "#D9D0C8",
        "axes.labelcolor": "#2C2420",
        "axes.titlecolor": "#2C2420",
        "xtick.color": "#2C2420",
        "ytick.color": "#2C2420",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    fig, ax = plt.subplots(figsize=(9.6, 5.35), dpi=150)
    _plot_curve(ax, x, worst_arr,   "Least relevant first", "#9E5C5C", (0, (1.5, 2.0)), 2)
    _plot_curve(ax, x, random_arr,  "Random order",         "#A8A09A", "--",             4)
    _plot_curve(ax, x, attnlrp_arr, "Most relevant first",  "#6B8CAE", "-",              6)

    ax.set_title(
        "Faithfulness Evaluation: Token Mask",
        fontsize=15.5,
        fontweight="semibold",
        loc="center",
        pad=17,
    )
    ax.set_xlabel("Tokens Masked (%)", fontsize=10.8, labelpad=9)
    ax.set_ylabel("Confidence predicted", fontsize=10.8, labelpad=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#E5E7EB", linewidth=1.0)
    ax.grid(axis="x", color="#F3F4F6", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9.8)
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#DEE2E6",
        fontsize=9.4,
        handlelength=3.2,
        borderpad=0.8,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "faithfulness_curve.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "faithfulness_curve.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "faithfulness_curve.svg", bbox_inches="tight")
    plt.close(fig)
    print("Saved: aiaa4051/task1/faithfulness/faithfulness_curve.png")
    print("Saved: aiaa4051/task1/faithfulness/faithfulness_curve.pdf")
    print("Saved: aiaa4051/task1/faithfulness/faithfulness_curve.svg")

    print(f"\nNormalized AUC:")
    print(f"  AttnLRP (relevance order) : {auc['attnlrp']:.4f}")
    print(f"  Random baseline           : {auc['random']:.4f}")
    print(f"  Least relevant baseline   : {auc['worst']:.4f}")
    print("(Lower AUC = faster confidence drop = more faithful explanation)")


if __name__ == "__main__":
    main()
