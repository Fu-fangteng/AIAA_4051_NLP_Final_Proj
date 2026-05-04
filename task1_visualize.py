"""
Task 1-C: Visualize Token Relevance
Multi-row wrapped layout — each row has TOKENS_PER_ROW tokens.
Run locally on Mac. Expected time: ~5 min.
"""

import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np

with open("aiaa4051/task1/relevance/relevances.pkl", "rb") as f:
    data = pickle.load(f)

print(f"Loaded {len(data)} samples.")

TOKENS_PER_ROW = 16   # tokens per row
ROW_HEIGHT     = 0.7  # inches per row
FONT_SIZE      = 10


def plot_relevance(sample_idx=0):
    sample  = data[sample_idx]
    tokens  = [t.replace("Ġ", " ").replace("Ċ", "↵") for t in sample["tokens"]]
    scores  = sample["relevance"].astype(float)

    # Diverging normalization: center at 0, scale by max abs
    # Positive = token promotes prediction, negative = suppresses
    abs_max = np.abs(scores).max() + 1e-8
    norm_scores = scores / abs_max           # in [-1, 1]
    # Map [-1,1] → [0,1] for colormap
    cmap_scores = (norm_scores + 1) / 2

    cmap   = plt.cm.RdYlGn
    colors = cmap(cmap_scores)

    n      = len(tokens)
    n_rows = (n + TOKENS_PER_ROW - 1) // TOKENS_PER_ROW

    fig_w  = TOKENS_PER_ROW * 0.95 + 0.5
    fig_h  = n_rows * ROW_HEIGHT + 1.2  # extra for title + colorbar

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, TOKENS_PER_ROW)
    ax.set_ylim(-0.5, n_rows + 0.3)
    ax.axis("off")
    ax.set_title(
        f"AttnLRP Token Relevance — Sample {sample_idx}\n"
        f"({n} tokens,  green = high relevance,  red = low / suppressive)",
        fontsize=11, pad=8
    )

    for idx, (tok, color) in enumerate(zip(tokens, colors)):
        row = idx // TOKENS_PER_ROW
        col = idx  % TOKENS_PER_ROW
        # Rows drawn top-to-bottom
        y = n_rows - 1 - row

        ax.text(
            col + 0.5, y + 0.5, tok,
            ha="center", va="center",
            fontsize=FONT_SIZE,
            bbox=dict(
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
                boxstyle=f"round,pad=0.3",
            ),
            clip_on=True,
        )

    # Colorbar
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, orientation="horizontal",
                      fraction=0.03, pad=0.02, aspect=40)
    cb.set_label("Relevance (neg → suppress,  pos → promote)", fontsize=9)
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])

    plt.tight_layout()
    out_path = f"aiaa4051/task1/relevance/sample_{sample_idx}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


for i in range(len(data)):
    plot_relevance(i)

print("All visualizations saved.")
