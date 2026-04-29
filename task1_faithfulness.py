"""
Task 1-D: Faithfulness Evaluation via Pixel Flipping
Mask tokens in order of decreasing relevance and record model confidence drop.
A faster confidence drop = higher faithfulness (explanation is more accurate).
- GPU recommended for N=200; local Mac feasible for N=20.
Expected time: 2-4 h on GPU.
"""

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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

def pixel_flipping(sample):
    """
    Progressively mask tokens by descending relevance order.
    Returns a list of model confidence values (length = seq_len).
    """
    tokens = list(sample["input_ids"])   # list of int
    scores = sample["relevance"]          # numpy array, shape [seq_len]
    sorted_idx = np.argsort(scores)[::-1].tolist()  # most → least important

    masked_tokens = tokens.copy()
    confidence_curve = []

    for idx in sorted_idx:
        masked_tokens[idx] = MASK_TOKEN_ID
        input_ids = torch.tensor([masked_tokens]).to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits
            prob = torch.softmax(logits[0, -1], dim=-1).max().item()
        confidence_curve.append(prob)

    return confidence_curve

# Also compute a random-order baseline for comparison
def random_baseline(sample):
    tokens = list(sample["input_ids"])
    seq_len = len(tokens)
    shuffled_idx = np.random.permutation(seq_len).tolist()

    masked_tokens = tokens.copy()
    confidence_curve = []

    for idx in shuffled_idx:
        masked_tokens[idx] = MASK_TOKEN_ID
        input_ids = torch.tensor([masked_tokens]).to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits
            prob = torch.softmax(logits[0, -1], dim=-1).max().item()
        confidence_curve.append(prob)

    return confidence_curve

np.random.seed(42)
attnlrp_curves = []
random_curves = []

for i, sample in enumerate(data):
    curve = pixel_flipping(sample)
    rand_curve = random_baseline(sample)
    if len(curve) > 0:
        attnlrp_curves.append(curve)
        random_curves.append(rand_curve)
    if (i + 1) % 10 == 0:
        print(f"Evaluated {i+1}/{len(data)} samples")

# Pad to same length for averaging
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

mean_attnlrp = attnlrp_arr.mean(axis=0)
mean_random  = random_arr.mean(axis=0)

# Plot
plt.figure(figsize=(9, 5))
x = np.linspace(0, 100, max_len)
plt.plot(x, mean_attnlrp, label="AttnLRP (relevance order)", color="steelblue", linewidth=2)
plt.plot(x, mean_random,  label="Random baseline",           color="gray",      linewidth=2, linestyle="--")
plt.xlabel("Tokens masked (%)")
plt.ylabel("Average model confidence")
plt.title("Faithfulness Evaluation — Pixel Flipping")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("aiaa4051/task1/faithfulness/faithfulness_curve.png", dpi=150)
print("Saved: aiaa4051/task1/faithfulness/faithfulness_curve.png")

auc_attnlrp = np.trapezoid(mean_attnlrp) / max_len
auc_random  = np.trapezoid(mean_random)  / max_len
print(f"\nNormalized AUC — AttnLRP: {auc_attnlrp:.4f}  |  Random: {auc_random:.4f}")
print("(Lower AUC = faster confidence drop = more faithful explanation)")

# Save curves for later analysis
with open("aiaa4051/task1/faithfulness/faithfulness_data.pkl", "wb") as f:
    pickle.dump({"attnlrp": attnlrp_arr, "random": random_arr,
                 "mean_attnlrp": mean_attnlrp, "mean_random": mean_random}, f)
print("Saved raw curve data to aiaa4051/task1/faithfulness/faithfulness_data.pkl")
