"""
Task 1-D: Faithfulness Evaluation via token deletion.

Computes three deletion-order curves and saves results to
`aiaa4051/task1/faithfulness/faithfulness_data.pkl`.
Run task1_visualization.py afterwards to render the figure.

  - AttnLRP order        : delete tokens by descending signed relevance
  - Random baseline      : delete tokens in random order
  - Low relevance first  : delete tokens by smallest absolute relevance first

Lower AUC = confidence drops faster = more faithful explanation.
Deletion is implemented with attention_mask=0, so the original token ids are
not replaced by GPT-2's EOS token.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from training_config import base_model_path, int_env

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def build_cumulative_attention_masks(seq_len, sorted_idx):
    if len(sorted_idx) != seq_len:
        raise ValueError("sorted_idx must contain exactly one index per token")

    attention_mask = [1] * seq_len
    masks = [attention_mask.copy()]
    for idx in sorted_idx:
        attention_mask[idx] = 0
        masks.append(attention_mask.copy())
    return masks


def mask_percent_axis(num_tokens):
    return np.linspace(0, 100, num_tokens + 1)


def relevance_first_order(scores):
    scores = np.asarray(scores)
    return np.argsort(scores)[::-1].tolist()


def least_relevant_first_order(scores):
    scores = np.asarray(scores)
    return np.argsort(np.abs(scores)).tolist()


def pad_curve(curves, length):
    padded = []
    for c in curves:
        if len(c) < length:
            c = c + [c[-1]] * (length - len(c))
        padded.append(c[:length])
    return np.array(padded)


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

max_len = max(len(c) for c in attnlrp_curves)

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

print(f"\nNormalized AUC:")
print(f"  AttnLRP (relevance order) : {auc_attnlrp:.4f}")
print(f"  Random baseline           : {auc_random:.4f}")
print(f"  Least relevant baseline   : {auc_worst:.4f}")
print("(Lower AUC = faster confidence drop = more faithful explanation)")

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
print("Saved: aiaa4051/task1/faithfulness/faithfulness_data.pkl")
