"""
Task 2-B: Pre-FT vs Post-FT AttnLRP Comparison
Compare token relevance distributions before and after sequential fine-tuning.
Runs on Mac (CPU). Expected time: 1-2 h.

Requires:
  - aiaa4051/data/squad_v2_dev200        (from task1_data_prep.py)
  - aiaa4051/checkpoints/sft_final       (from task2_sft.py)
  - lxt_patch.py                         (shared CP-LRP helper)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from transformers import GPT2Tokenizer
from datasets import load_from_disk

from lxt_patch import apply_gpt2_cplrp, get_relevance, load_lrp_model
from training_config import base_model_path, get_device

# Apply patches once before loading any model
apply_gpt2_cplrp(verbose=True)

BASE_MODEL = base_model_path()

tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

device = get_device()
print(f"Using device: {device}")

model_pre  = load_lrp_model(BASE_MODEL,                       device)
model_post = load_lrp_model("aiaa4051/checkpoints/sft_final", device)

dataset = load_from_disk("aiaa4051/data/squad_v2_dev200")
N = 20

results = []
for i in range(N):
    sample    = dataset[i]
    input_ids = sample["input_ids"]
    tokens    = tokenizer.convert_ids_to_tokens(input_ids)

    r_pre  = get_relevance(model_pre,  input_ids, device)
    r_post = get_relevance(model_post, input_ids, device)
    delta  = r_post - r_pre

    results.append({"tokens": tokens, "pre": r_pre, "post": r_post, "delta": delta})
    if (i + 1) % 5 == 0:
        print(f"Compared {i+1}/{N}")

with open("aiaa4051/task2/comparison/comparison.pkl", "wb") as f:
    pickle.dump(results, f)
print("Saved comparison data.")

# ── Visualization ─────────────────────────────────────────────────────────────
TOKENS_PER_ROW = 16

def plot_comparison(sample_idx=0):
    r = results[sample_idx]
    tokens = [t.replace("Ġ", " ").replace("Ċ", "↵") for t in r["tokens"]]
    n = len(tokens)
    n_rows = (n + TOKENS_PER_ROW - 1) // TOKENS_PER_ROW

    fig, axes = plt.subplots(3, 1, figsize=(TOKENS_PER_ROW * 0.95 + 0.5,
                                             n_rows * 0.65 * 3 + 1.5))
    fig.suptitle(f"Pre vs Post Fine-Tuning AttnLRP — Sample {sample_idx}", fontsize=12)

    panels = [
        (r["pre"],   "RdYlGn", "Pre-FT Relevance"),
        (r["post"],  "RdYlGn", "Post-FT Relevance"),
        (r["delta"], "RdBu",   "Delta (Post − Pre)"),
    ]

    for ax, (vals, cmap_name, title) in zip(axes, panels):
        abs_max = np.abs(vals).max() + 1e-8
        norm_v  = vals / abs_max
        cmap_v  = (norm_v + 1) / 2
        colors  = plt.get_cmap(cmap_name)(cmap_v)

        ax.set_xlim(0, TOKENS_PER_ROW)
        ax.set_ylim(-0.1, n_rows + 0.1)
        ax.axis("off")
        ax.set_title(title, fontsize=10, loc="left", pad=4)

        for idx, (tok, color) in enumerate(zip(tokens, colors)):
            row = idx // TOKENS_PER_ROW
            col = idx  % TOKENS_PER_ROW
            y   = n_rows - 1 - row
            ax.text(col + 0.5, y + 0.5, tok,
                    ha="center", va="center", fontsize=9,
                    bbox=dict(facecolor=color, edgecolor="white",
                              linewidth=0.4, boxstyle="round,pad=0.3"))

        sm = plt.cm.ScalarMappable(cmap=cmap_name,
                                   norm=mcolors.Normalize(vmin=-1, vmax=1))
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, orientation="vertical",
                          fraction=0.015, pad=0.01)
        cb.set_ticks([-1, 0, 1])
        cb.ax.tick_params(labelsize=8)

    plt.tight_layout()
    out = f"aiaa4051/task2/comparison/compare_sample{sample_idx}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

for i in range(N):
    plot_comparison(i)

# ── Summary: mean delta per token position ───────────────────────────────────
min_len = min(len(r["delta"]) for r in results)
delta_mat = np.array([r["delta"][:min_len] for r in results])
mean_delta = delta_mat.mean(axis=0)

plt.figure(figsize=(12, 3))
plt.bar(range(min_len), mean_delta,
        color=["#d73027" if v < 0 else "#1a9850" for v in mean_delta])
plt.axhline(0, color="black", linewidth=0.8)
plt.xlabel("Token position")
plt.ylabel("Mean delta relevance")
plt.title(f"Mean Relevance Change After SFT (N={N} samples)")
plt.tight_layout()
plt.savefig("aiaa4051/task2/comparison/mean_delta.png", dpi=150)
print("Saved: aiaa4051/task2/comparison/mean_delta.png")
