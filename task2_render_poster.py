"""
Render a poster-friendly Task 2 question-only word-level heatmap.

This script is render-only: it reads cached AttnLRP results from
`aiaa4051/task2/comparison/comparison.pkl` and does not load GPT-2.
"""

import argparse
import pickle
import re
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk

from task2_poster_utils import aggregate_question_words


ROOT = Path(__file__).resolve().parent
DEFAULT_SAMPLE_IDX = 3
OUT_DIR = ROOT / "aiaa4051" / "task2" / "comparison"
COMPARISON_PATH = OUT_DIR / "comparison.pkl"
DATASET_PATH = ROOT / "aiaa4051" / "data" / "squad_v2_dev200"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-idx", type=int, default=DEFAULT_SAMPLE_IDX)
    args = parser.parse_args()

    dataset = load_from_disk(str(DATASET_PATH))
    with open(COMPARISON_PATH, "rb") as f:
        comparison = pickle.load(f)

    sample_idx = args.sample_idx
    sample_text = dataset[sample_idx]["text"]
    question = _extract_question(sample_text)
    context = _context_excerpt(sample_text)
    record = comparison[sample_idx]

    words, pre_signed = aggregate_question_words(record["tokens"], record["pre"])
    post_words, post_signed = aggregate_question_words(record["tokens"], record["post"])
    if words != post_words:
        raise ValueError("Pre- and post-FT question words do not align")

    pre_importance = np.abs(pre_signed)
    post_importance = np.abs(post_signed)
    delta_importance = post_importance - pre_importance

    out_base = OUT_DIR / f"poster_sample{sample_idx}"
    render_question_heatmap(
        words=words,
        pre_importance=pre_importance,
        post_importance=post_importance,
        delta_importance=delta_importance,
        question=question,
        context=context,
        sample_idx=sample_idx,
        out_base=out_base,
    )


def render_question_heatmap(
    words,
    pre_importance,
    post_importance,
    delta_importance,
    question,
    context,
    sample_idx,
    out_base,
):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#E5E7EB",
        "axes.labelcolor": "#111827",
        "xtick.color": "#111827",
        "ytick.color": "#111827",
    })

    fig = plt.figure(figsize=(10.8, 5.8), dpi=160)
    gs = fig.add_gridspec(
        nrows=4,
        ncols=1,
        height_ratios=[0.92, 1.0, 0.48, 0.42],
        hspace=0.32,
    )

    title_ax = fig.add_subplot(gs[0, 0])
    title_ax.axis("off")
    title_ax.text(
        0,
        0.92,
        "Fine-Tuning Changes Question-Word Importance",
        fontsize=18,
        fontweight="semibold",
        color="#111827",
        va="top",
    )
    title_ax.text(
        0,
        0.48,
        f'Question: "{question}"',
        fontsize=12.5,
        color="#111827",
        va="top",
    )
    title_ax.text(
        0,
        0.13,
        f"Context excerpt: {context}",
        fontsize=9.8,
        color="#4B5563",
        va="top",
    )

    importance = np.vstack([pre_importance, post_importance])
    vmax_importance = max(float(np.max(importance)), 1e-8)
    imp_ax = fig.add_subplot(gs[1, 0])
    imp_norm = mcolors.Normalize(vmin=0, vmax=vmax_importance)
    imp_ax.imshow(importance, cmap="YlGnBu", norm=imp_norm, aspect="auto")
    _style_heatmap_axis(imp_ax, words, ["Pre-FT", "Post-FT"], show_words=True)
    _annotate_cells(imp_ax, importance, imp_norm, "{:.1f}")

    delta_ax = fig.add_subplot(gs[2, 0])
    vmax_delta = max(float(np.max(np.abs(delta_importance))), 1e-8)
    delta_norm = mcolors.TwoSlopeNorm(vmin=-vmax_delta, vcenter=0, vmax=vmax_delta)
    delta_ax.imshow(
        delta_importance[np.newaxis, :],
        cmap="RdBu",
        norm=delta_norm,
        aspect="auto",
    )
    _style_heatmap_axis(delta_ax, words, ["Post - Pre"], show_words=False)
    _annotate_cells(delta_ax, delta_importance[np.newaxis, :], delta_norm, "{:+.1f}")

    note_ax = fig.add_subplot(gs[3, :])
    note_ax.axis("off")
    note_ax.text(
        0,
        0.72,
        "GPT-2 subword relevance scores are summed into word-level values. "
        "Top rows: darker blue-green = stronger |AttnLRP|. Bottom row: blue = increased importance, red = decreased.",
        fontsize=9.2,
        color="#4B5563",
        va="center",
    )
    note_ax.text(
        0,
        0.18,
        f"Sample {sample_idx}; render-only from cached comparison.pkl",
        fontsize=8.5,
        color="#6B7280",
        va="center",
    )

    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_base}.png")
    print(f"Saved: {out_base}.pdf")


def _style_heatmap_axis(ax, words, row_labels, show_words):
    ax.set_xticks(np.arange(len(words)))
    ax.set_xticklabels(words if show_words else [], fontsize=11.5)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=11.5)
    ax.tick_params(
        top=show_words,
        bottom=False,
        labeltop=show_words,
        labelbottom=False,
        length=0,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(words), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2.4)
    ax.tick_params(which="minor", bottom=False, left=False)


def _annotate_cells(ax, values, norm, fmt):
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            val = float(values[row, col])
            rgba = ax.images[0].cmap(norm(val))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            color = "white" if luminance < 0.45 else "#111827"
            ax.text(col, row, fmt.format(val), ha="center", va="center", fontsize=10, color=color)


def _extract_question(text):
    match = re.search(r"Question:\s*(.*?)\nAnswer:", text, flags=re.S)
    if not match:
        raise ValueError("Question field not found")
    return match.group(1).strip()


def _context_excerpt(text):
    context = text.split("Question:", 1)[0].replace("Context:", "", 1).strip()
    if "descended from Norse" in context:
        return "The Normans ... descended from Norse raiders from Denmark, Iceland and Norway ..."
    if len(context) <= 145:
        return context
    return context[:142].rstrip() + "..."


if __name__ == "__main__":
    main()
