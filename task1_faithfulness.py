"""
Task 1-D: Faithfulness Evaluation via Pixel Flipping
Three curves are plotted:
  - AttnLRP order   : mask tokens by descending relevance (most important first)
  - Random baseline : mask tokens in random order
  - Worst order     : mask tokens by ascending relevance (least important first)

Lower AUC = confidence drops faster = more faithful explanation.
AttnLRP should sit between worst (upper bound) and random.

Note: CP-LRP scores are signed (positive = promotes, negative = suppresses).
Masking high-positive tokens first can cause a non-monotonic curve once
masking crosses into suppressive tokens (whose removal raises confidence).
The worst-order baseline bounds this effect from above.

GPU recommended for N=200; local Mac feasible for N=20.
Expected time: 2-4 h on GPU.
"""

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# np.trapezoid was introduced in NumPy 2.0; fall back to np.trapz on older versions
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

tokenizer = GPT2Tokenizer.from_pretrained("/home/user/project/gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("/home/user/project/gpt2")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

with open("aiaa4051/task1/relevance/relevances.pkl", "rb") as f:
    data = pickle.load(f)

print(f"Loaded {len(data)} samples for faithfulness evaluation.")

# GPT-2 uses eos_token_id (50256) as the mask/pad token
MASK_TOKEN_ID = tokenizer.eos_token_id


def _run_curve(tokens, sorted_idx):
    """Progressively mask tokens in sorted_idx order; return confidence curve."""
    masked_tokens = list(tokens)
    curve = []
    for idx in sorted_idx:
        masked_tokens[idx] = MASK_TOKEN_ID
        input_ids = torch.tensor([masked_tokens]).to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits
            prob = torch.softmax(logits[0, -1], dim=-1).max().item()
        curve.append(prob)
    return curve


def pixel_flipping(sample):
    """Mask tokens by descending relevance (most important first)."""
    scores = sample["relevance"]
    return _run_curve(sample["input_ids"], np.argsort(scores)[::-1].tolist())


def worst_order_baseline(sample):
    """Mask tokens by ascending relevance (least important first).
    Provides the theoretical upper-bound AUC for this metric."""
    scores = sample["relevance"]
    return _run_curve(sample["input_ids"], np.argsort(scores).tolist())


def random_baseline(sample):
    """Mask tokens in random order."""
    seq_len = len(sample["input_ids"])
    return _run_curve(sample["input_ids"], np.random.permutation(seq_len).tolist())


np.random.seed(42)
attnlrp_curves = []
random_curves  = []
worst_curves   = []

for i, sample in enumerate(data):
    curve       = pixel_flipping(sample)
    rand_curve  = random_baseline(sample)
    worst_curve = worst_order_baseline(sample)

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

# Plot all three curves
plt.figure(figsize=(9, 5))
x = np.linspace(0, 100, max_len)
plt.plot(x, mean_attnlrp, label="AttnLRP (relevance order)",        color="steelblue", linewidth=2)
plt.plot(x, mean_random,  label="Random baseline",                  color="gray",      linewidth=2, linestyle="--")
plt.plot(x, mean_worst,   label="Worst order (ascending, upper bd)", color="tomato",    linewidth=2, linestyle=":")
plt.xlabel("Tokens masked (%)")
plt.ylabel("Average model confidence")
plt.title("Faithfulness Evaluation — Pixel Flipping\n"
          "(lower AUC = faster confidence drop = more faithful)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("aiaa4051/task1/faithfulness/faithfulness_curve.png", dpi=150)
print("Saved: aiaa4051/task1/faithfulness/faithfulness_curve.png")

auc_attnlrp = _trapz(mean_attnlrp) / max_len
auc_random  = _trapz(mean_random)  / max_len
auc_worst   = _trapz(mean_worst)   / max_len
print(f"\nNormalized AUC:")
print(f"  AttnLRP (relevance order) : {auc_attnlrp:.4f}")
print(f"  Random baseline           : {auc_random:.4f}")
print(f"  Worst order (upper bound) : {auc_worst:.4f}")
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
    }, f)
print("Saved raw curve data to aiaa4051/task1/faithfulness/faithfulness_data.pkl")
