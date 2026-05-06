"""
Render Task 3 poster figure options from cached parameter relevance data.

This script is render-only: it reads
`aiaa4051/task3/comparison/param_relevance_data.pkl` and does not load models.
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def build_param_relevance_payload(rel_a, rel_b, test_texts_a, test_texts_b):
    rel_a = np.asarray(rel_a, dtype=float)
    rel_b = np.asarray(rel_b, dtype=float)
    if rel_a.shape != rel_b.shape:
        raise ValueError("rel_a and rel_b must have the same shape")

    rel_a_norm = rel_a / (rel_a.sum() + 1e-8)
    rel_b_norm = rel_b / (rel_b.sum() + 1e-8)

    return {
        "rel_A": rel_a,
        "rel_B": rel_b,
        "rel_A_norm": rel_a_norm,
        "rel_B_norm": rel_b_norm,
        "diff_norm": rel_a_norm - rel_b_norm,
        "layers": np.arange(len(rel_a)),
        "test_texts_A": list(test_texts_a),
        "test_texts_B": list(test_texts_b),
        "model_A": "SciQ",
        "model_B": "SQuAD_v2",
        "metric": "activation_magnitude_proxy",
    }


def top_abs_difference_layers(diff, n=3):
    diff = np.asarray(diff, dtype=float)
    return np.argsort(np.abs(diff))[::-1][:n]


ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "aiaa4051" / "task3" / "comparison"
DATA_PATH = OUT_DIR / "param_relevance_data.pkl"

# Modern color palette
COLOR_A = "#6366F1"  # Indigo - Professional
COLOR_B = "#F59E0B"  # Amber - Warm
COLOR_POS = "#10B981"  # Emerald - Positive
COLOR_NEG = "#EF4444"  # Red - Negative
TEXT = "#1F2937"  # Dark gray for text
MUTED = "#6B7280"  # Medium gray for secondary text
GRID = "#F3F4F6"  # Light gray for grid
BG = "#FAFBFC"  # Very light background


def main():
    payload = load_payload()
    render_bars_model_a(payload)
    render_bars_model_b(payload)
    render_bars_difference(payload)
    render_comparison_diverging(payload)  # 方案A: Split Diverging
    render_comparison_grouped(payload)    # 方案B: Improved Grouped


def load_payload():
    with open(DATA_PATH, "rb") as f:
        return pickle.load(f)


def render_bars_model_a(payload):
    """Model A (SciQ) parameter relevance bar chart."""
    layers = payload["layers"]
    rel_a = payload["rel_A_norm"]

    _set_style()
    fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=160)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor("white")

    bars = ax.bar(layers, rel_a, color=COLOR_A, width=0.7, edgecolor="none", alpha=0.88)

    ax.set_ylabel("Normalized relevance", fontsize=11.5, fontweight="500", color=TEXT)
    ax.set_xlabel("Transformer layer", fontsize=11.5, fontweight="500", color=TEXT)
    ax.set_title("Model A: SciQ Fine-tuned Model", fontsize=15, fontweight="bold", pad=20, color=TEXT, loc="left")
    ax.text(0, 1.07, "Parameter importance distribution across layers",
            transform=ax.transAxes, fontsize=11, color=MUTED, style="italic")

    ax.set_xticks(layers)
    ax.set_xticklabels([str(int(x)) for x in layers], fontsize=10.5)
    ax.set_ylim(0, max(rel_a) * 1.18)

    _clean_axis(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.75, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout(pad=2.0)
    _save(fig, "task3_model_a_sciq")


def render_bars_model_b(payload):
    """Model B (SQuAD_v2) parameter relevance bar chart."""
    layers = payload["layers"]
    rel_b = payload["rel_B_norm"]

    _set_style()
    fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=160)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor("white")

    bars = ax.bar(layers, rel_b, color=COLOR_B, width=0.7, edgecolor="none", alpha=0.88)

    ax.set_ylabel("Normalized relevance", fontsize=11.5, fontweight="500", color=TEXT)
    ax.set_xlabel("Transformer layer", fontsize=11.5, fontweight="500", color=TEXT)
    ax.set_title("Model B: SQuAD_v2 Fine-tuned Model", fontsize=15, fontweight="bold", pad=20, color=TEXT, loc="left")
    ax.text(0, 1.07, "Parameter importance distribution across layers",
            transform=ax.transAxes, fontsize=11, color=MUTED, style="italic")

    ax.set_xticks(layers)
    ax.set_xticklabels([str(int(x)) for x in layers], fontsize=10.5)
    ax.set_ylim(0, max(rel_b) * 1.18)

    _clean_axis(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.75, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout(pad=2.0)
    _save(fig, "task3_model_b_squad")


def render_bars_difference(payload):
    """Difference in parameter relevance between models."""
    layers = payload["layers"]
    diff = payload["diff_norm"]

    _set_style()
    fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=160)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor("white")

    colors = [COLOR_POS if x > 0 else COLOR_NEG for x in diff]
    bars = ax.bar(layers, diff, color=colors, width=0.7, edgecolor="none", alpha=0.85)

    # Zero line
    ax.axhline(0, color=TEXT, linewidth=1.1, alpha=0.25, zorder=1)

    ax.set_ylabel("Difference in relevance (SciQ - SQuAD_v2)", fontsize=11.5, fontweight="500", color=TEXT)
    ax.set_xlabel("Transformer layer", fontsize=11.5, fontweight="500", color=TEXT)
    ax.set_title("Model Comparison: Which Layers Shift?", fontsize=15, fontweight="bold", pad=20, color=TEXT, loc="left")
    ax.text(0, 1.07, "Green: SciQ dominates  |  Red: SQuAD_v2 dominates",
            transform=ax.transAxes, fontsize=11, color=MUTED)

    ax.set_xticks(layers)
    ax.set_xticklabels([str(int(x)) for x in layers], fontsize=10.5)

    _clean_axis(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.75, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    _add_top_layer_note(ax, payload, y=-0.15)

    fig.tight_layout(pad=2.0)
    _save(fig, "task3_model_comparison_diff")


def _add_top_layer_note(ax, payload, y=0.9):
    diff = payload["diff_norm"]
    top = top_abs_difference_layers(diff, n=3)
    parts = []
    for layer in top:
        who = "SciQ" if diff[layer] > 0 else "SQuAD"
        parts.append(f"L{int(layer)}: {who} {diff[layer]:+.3f}")
    ax.text(
        0.01,
        y,
        "Largest shifts: " + " | ".join(parts),
        transform=ax.transAxes,
        fontsize=9.5,
        color=MUTED,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=GRID, linewidth=0.8),
    )


def _set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": BG,
        "axes.facecolor": "white",
        "axes.edgecolor": "none",
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
    })


def _clean_axis(ax):
    """Clean up axis appearance."""
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="both", labelsize=10.5, colors=TEXT)


def _save(fig, name):
    """Save figure as PNG only."""
    png_path = OUT_DIR / f"{name}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"✓ {name}.png")


def render_comparison_diverging(payload):
    """Plan A: Split Diverging Bar Chart - SciQ vs SQuAD side by side."""
    layers = payload["layers"]
    rel_a = payload["rel_A_norm"]
    rel_b = payload["rel_B_norm"]

    _set_style()
    fig, ax = plt.subplots(figsize=(11.5, 7.0), dpi=160)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor("white")

    y_pos = np.arange(len(layers))

    # SciQ bars go left (negative)
    ax.barh(y_pos, -rel_a, height=0.65, label="Model A: SciQ",
            color=COLOR_A, edgecolor="none", alpha=0.88)

    # SQuAD bars go right (positive)
    ax.barh(y_pos, rel_b, height=0.65, label="Model B: SQuAD_v2",
            color=COLOR_B, edgecolor="none", alpha=0.88)

    # Center line
    ax.axvline(0, color=TEXT, linewidth=1.2, alpha=0.3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Layer {int(x)}" for x in layers], fontsize=10.5)
    ax.set_xlabel("Normalized parameter relevance", fontsize=12, fontweight="500", color=TEXT)
    ax.set_title("Plan A: Model Comparison - Diverging View", fontsize=15, fontweight="bold", pad=22, color=TEXT, loc="left")
    ax.text(0, 1.05, "SciQ (left) vs SQuAD_v2 (right) - wider bar = higher importance",
            transform=ax.transAxes, fontsize=11, color=MUTED, style="italic")

    ax.legend(loc="lower right", frameon=False, fontsize=11, labelspacing=1.2)
    ax.grid(axis="x", color=GRID, linewidth=0.75, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    _clean_axis(ax)

    fig.tight_layout(pad=2.0)
    _save(fig, "task3_comparison_plan_a_diverging")


def render_comparison_grouped(payload):
    """Plan B: Improved Grouped Bar Chart with better styling."""
    layers = payload["layers"]
    rel_a = payload["rel_A_norm"]
    rel_b = payload["rel_B_norm"]

    _set_style()
    fig, ax = plt.subplots(figsize=(12.0, 6.8), dpi=160)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor("white")

    x = np.arange(len(layers))
    width = 0.38

    # Bars with gradient-like effect (slight shadow)
    bars1 = ax.bar(x - width/2, rel_a, width, label="Model A: SciQ",
                   color=COLOR_A, edgecolor="none", alpha=0.88)
    bars2 = ax.bar(x + width/2, rel_b, width, label="Model B: SQuAD_v2",
                   color=COLOR_B, edgecolor="none", alpha=0.88)

    # Add subtle value labels on top of bars (for highest values)
    for i, (v1, v2) in enumerate(zip(rel_a, rel_b)):
        max_val = max(v1, v2)
        if max_val > np.percentile([rel_a, rel_b], 75):  # Only top 25%
            if v1 > v2:
                ax.text(i - width/2, v1 + 0.005, f"{v1:.2f}",
                       ha="center", va="bottom", fontsize=8.5, color=COLOR_A, fontweight="500")
            else:
                ax.text(i + width/2, v2 + 0.005, f"{v2:.2f}",
                       ha="center", va="bottom", fontsize=8.5, color=COLOR_B, fontweight="500")

    ax.set_ylabel("Normalized parameter relevance", fontsize=12, fontweight="500", color=TEXT)
    ax.set_xlabel("Transformer layer", fontsize=12, fontweight="500", color=TEXT)
    ax.set_title("Plan B: Model Comparison - Grouped View", fontsize=15, fontweight="bold", pad=22, color=TEXT, loc="left")
    ax.text(0, 1.05, "Side-by-side comparison - direct value inspection",
            transform=ax.transAxes, fontsize=11, color=MUTED, style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels([f"L{int(i)}" for i in layers], fontsize=10.5)
    ax.set_ylim(0, max(max(rel_a), max(rel_b)) * 1.15)

    ax.legend(loc="upper right", frameon=False, fontsize=11, labelspacing=1.2)
    ax.grid(axis="y", color=GRID, linewidth=0.75, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    _clean_axis(ax)

    fig.tight_layout(pad=2.0)
    _save(fig, "task3_comparison_plan_b_grouped")


if __name__ == "__main__":
    main()
