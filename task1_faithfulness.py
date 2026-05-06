"""
Task 1-D: Faithfulness Evaluation via token deletion.

Three curves are plotted:
  - AttnLRP order        : delete tokens by descending signed relevance
  - Random baseline      : delete tokens in random order
  - Low relevance first  : delete tokens by smallest absolute relevance first

Lower AUC = confidence drops faster = more faithful explanation.
Deletion is implemented with attention_mask=0, so the original token ids are
not replaced by GPT-2's EOS token.
"""

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from faithfulness_utils import (
    build_cumulative_attention_masks,
    least_relevant_first_order,
    mask_percent_axis,
    relevance_first_order,
)
from training_config import base_model_path, int_env

# np.trapezoid was introduced in NumPy 2.0; fall back to np.trapz on older versions
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

BASE_MODEL = base_model_path()
FAITHFULNESS_BATCH_SIZE = int_env("FAITHFULNESS_BATCH_SIZE", 32)

tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(BASE_MODEL)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")
print(f"Faithfulness batch size: {FAITHFULNESS_BATCH_SIZE}")

with open("aiaa4051/task1/relevance/relevances.pkl", "rb") as f:
    data = pickle.load(f)

print(f"Loaded {len(data)} samples for faithfulness evaluation.")


def _reference_prediction(tokens):
    input_ids = torch.tensor([tokens], device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(logits[0, -1], dim=-1)
    target_token_id = int(probs.argmax().item())
    baseline_prob = float(probs[target_token_id].item())
    return target_token_id, baseline_prob


def _run_curve(tokens, sorted_idx, target_token_id, baseline_prob):
    """Progressively delete tokens and track the original predicted token."""
    # Drop the first unmasked entry here because baseline_prob is shared
    # across all deletion orders for the sample.
    attention_masks = build_cumulative_attention_masks(len(tokens), sorted_idx)[1:]
    token_ids = torch.tensor(tokens, device=device).unsqueeze(0)
    curve = [baseline_prob]
    for start in range(0, len(attention_masks), FAITHFULNESS_BATCH_SIZE):
        batch_masks = attention_masks[start:start + FAITHFULNESS_BATCH_SIZE]
        attention_mask = torch.tensor(batch_masks, device=device)
        input_ids = token_ids.expand(attention_mask.shape[0], -1)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits[:, -1], dim=-1)[:, target_token_id]
        curve.extend(probs.detach().cpu().tolist())
    return curve


def pixel_flipping(sample, target_token_id, baseline_prob):
    """Delete tokens by descending signed relevance."""
    scores = sample["relevance"]
    return _run_curve(
        sample["input_ids"],
        relevance_first_order(scores),
        target_token_id,
        baseline_prob,
    )


def worst_order_baseline(sample, target_token_id, baseline_prob):
    """Delete tokens by lowest absolute relevance first."""
    scores = sample["relevance"]
    return _run_curve(
        sample["input_ids"],
        least_relevant_first_order(scores),
        target_token_id,
        baseline_prob,
    )


def random_baseline(sample, target_token_id, baseline_prob):
    """Delete tokens in random order."""
    seq_len = len(sample["input_ids"])
    return _run_curve(
        sample["input_ids"],
        np.random.permutation(seq_len).tolist(),
        target_token_id,
        baseline_prob,
    )


np.random.seed(42)
attnlrp_curves = []
random_curves  = []
worst_curves   = []

for i, sample in enumerate(data):
    target_token_id, baseline_prob = _reference_prediction(sample["input_ids"])
    curve       = pixel_flipping(sample, target_token_id, baseline_prob)
    rand_curve  = random_baseline(sample, target_token_id, baseline_prob)
    worst_curve = worst_order_baseline(sample, target_token_id, baseline_prob)

    if len(curve) > 0:
        attnlrp_curves.append(curve)
        random_curves.append(rand_curve)
        worst_curves.append(worst_curve)

    if (i + 1) % 10 == 0:
        print(f"Evaluated {i+1}/{len(data)} samples")

# Pad all curves to the same length for averaging
max_len = max(len(c) for c in attnlrp_curves)


def pad_curve(curves, length):
    padded = []
    for c in curves:
        if len(c) < length:
            c = c + [c[-1]] * (length - len(c))
        padded.append(c[:length])
    return np.array(padded)


attnlrp_arr = pad_curve(attnlrp_curves, max_len)
random_arr  = pad_curve(random_curves,  max_len)
worst_arr   = pad_curve(worst_curves,   max_len)

mean_attnlrp = attnlrp_arr.mean(axis=0)
mean_random  = random_arr.mean(axis=0)
mean_worst   = worst_arr.mean(axis=0)

x = mask_percent_axis(max_len - 1)
auc_attnlrp = _trapz(mean_attnlrp, x) / 100
auc_random  = _trapz(mean_random, x)  / 100
auc_worst   = _trapz(mean_worst, x)   / 100


def _mean_and_sem(curves):
    mean = curves.mean(axis=0)
    if len(curves) <= 1:
        return mean, np.zeros_like(mean)
    sem = curves.std(axis=0, ddof=1) / np.sqrt(len(curves))
    return mean, sem


def _plot_curve(ax, x_axis, curves, label, color, linestyle, auc, zorder):
    mean, sem = _mean_and_sem(curves)
    ax.plot(
        x_axis,
        mean,
        label=f"{label}  AUC={auc:.3f}",
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


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#D1D5DB",
    "axes.labelcolor": "#111827",
    "axes.titlecolor": "#111827",
    "xtick.color": "#374151",
    "ytick.color": "#374151",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

fig, ax = plt.subplots(figsize=(9.6, 5.35), dpi=150)
_plot_curve(ax, x, worst_arr, "Low relevance first", "#F97316", (0, (1.5, 2.0)), auc_worst, 2)
_plot_curve(ax, x, random_arr, "Random", "#6B7280", "--", auc_random, 4)
_plot_curve(ax, x, attnlrp_arr, "AttnLRP", "#2563EB", "-", auc_attnlrp, 6)

ax.set_title(
    "Faithfulness Evaluation: Token Deletion",
    fontsize=15.5,
    fontweight="semibold",
    loc="left",
    pad=17,
)
ax.text(
    0.0,
    1.015,
    f"Lower AUC = faster confidence drop   |   N = {len(attnlrp_arr)}   |   deletion uses attention_mask=0",
    transform=ax.transAxes,
    fontsize=9.7,
    color="#4B5563",
)
ax.set_xlabel("Tokens deleted (%)", fontsize=10.8, labelpad=9)
ax.set_ylabel("Confidence in original predicted token", fontsize=10.8, labelpad=9)
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
    edgecolor="#E5E7EB",
    fontsize=9.4,
    handlelength=3.2,
    borderpad=0.8,
)
fig.tight_layout()
fig.savefig("aiaa4051/task1/faithfulness/faithfulness_curve.png", dpi=300, bbox_inches="tight")
fig.savefig("aiaa4051/task1/faithfulness/faithfulness_curve.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved: aiaa4051/task1/faithfulness/faithfulness_curve.png")
print("Saved: aiaa4051/task1/faithfulness/faithfulness_curve.pdf")

print(f"\nNormalized AUC:")
print(f"  AttnLRP (relevance order) : {auc_attnlrp:.4f}")
print(f"  Random baseline           : {auc_random:.4f}")
print(f"  Least relevant baseline   : {auc_worst:.4f}")
print("(Lower AUC = faster confidence drop = more faithful explanation)")

# Save curves for later analysis
with open("aiaa4051/task1/faithfulness/faithfulness_data.pkl", "wb") as f:
    pickle.dump({
        "attnlrp":      attnlrp_arr,
        "random":       random_arr,
        "worst":        worst_arr,
        "mean_attnlrp": mean_attnlrp,
        "mean_random":  mean_random,
        "mean_worst":   mean_worst,
        "x":            x,
        "auc": {
            "attnlrp": auc_attnlrp,
            "random":  auc_random,
            "worst":   auc_worst,
        },
        "deletion_method": "attention_mask=0",
    }, f)
print("Saved raw curve data to aiaa4051/task1/faithfulness/faithfulness_data.pkl")
