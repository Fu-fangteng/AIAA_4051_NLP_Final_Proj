"""
Task 3-B: Per-Layer Parameter Relevance Comparison (Model A vs Model B)
Two metrics are computed:
    1) Logits ablation (preferred): zero-out one layer's output and
         measure mean absolute logit drop.
    2) Activation-magnitude proxy (legacy):
         R_layer = |x_mean @ W|.sum()   (x_mean = mean over sequence positions)

Each model is evaluated on test inputs that match its own training format:
    - Model A (SciQ)    : "Question: ...\nAnswer:" (no Context prefix)
    - Model B (SQuAD_v2): "Context: ...\nQuestion: ...\nAnswer:"

Requires:
    - aiaa4051/task3/modelA/final   (from task3_train_models.py)
    - aiaa4051/task3/modelB/final   (from task3_train_models.py)

Run on Mac or GPU. Expected time: 20 min (CPU) / 5 min (GPU).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from transformers import GPT2Tokenizer

from lxt_patch import load_lrp_model
from task3_relevance_utils import build_param_relevance_payload
from training_config import base_model_path

tokenizer = GPT2Tokenizer.from_pretrained(base_model_path())
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Model-appropriate test inputs ─────────────────────────────────────────────
# Model A was trained on SciQ format (no Context prefix).
TEST_TEXTS_A = [
    "Question: What is water made of?\nAnswer:",
    "Question: What is the sun?\nAnswer:",
    "Question: How do plants make food?\nAnswer:",
]

# Model B was trained on SQuAD format (with Context prefix).
TEST_TEXTS_B = [
    "Context: Water is a chemical compound of hydrogen and oxygen.\nQuestion: What is water?\nAnswer:",
    "Context: The sun is a star at the center of the solar system.\nQuestion: What is the sun?\nAnswer:",
    "Context: Photosynthesis is the process by which plants make food.\nQuestion: How do plants make food?\nAnswer:",
]

# ── Per-layer parameter relevance ─────────────────────────────────────────────

def compute_param_relevance_proxy(model_path, input_texts):
    """
    Returns a numpy array of shape [n_layers] with the activation-magnitude
    proxy for each MLP c_fc layer, averaged over all input_texts.

    R_layer = |x_mean @ W|.sum()
    where x_mean is the mean hidden state over sequence positions.
    No gradient computation is needed — uses torch.no_grad() throughout.
    """
    model = load_lrp_model(model_path, device)
    all_per_layer = []

    for input_text in input_texts:
        input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"].to(device)

        layer_inputs = {}
        hooks = []
        for idx, block in enumerate(model.transformer.h):
            def make_hook(i):
                def hook_fn(module, inp, out):
                    layer_inputs[i] = inp[0].detach()
                return hook_fn
            hooks.append(block.mlp.c_fc.register_forward_hook(make_hook(idx)))

        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)

        for h in hooks:
            h.remove()

        per_layer = []
        for idx, block in enumerate(model.transformer.h):
            # GPT-2 Conv1D stores weight as (in_features, out_features) = (768, 3072)
            W = block.mlp.c_fc.weight.detach()   # [768, 3072]
            if idx in layer_inputs:
                x = layer_inputs[idx]            # [1, seq_len, 768]
                x_mean = x.mean(dim=1)           # [1, 768]
                R = (x_mean @ W).abs().sum().item()
            else:
                R = 0.0
            per_layer.append(R)

        all_per_layer.append(per_layer)

    return np.mean(all_per_layer, axis=0)


def compute_param_relevance_ablation(model_path, input_texts):
    """
    Returns a numpy array of shape [n_layers] with the mean absolute
    logit drop caused by zeroing a single layer's output (one layer at a time).

    This is a direct output-based contribution estimate and is preferred
    over the activation-magnitude proxy for fairness.
    """
    model = load_lrp_model(model_path, device)
    n_layers = len(model.transformer.h)
    all_per_layer = []

    for input_text in input_texts:
        input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"].to(device)

        with torch.no_grad():
            base_logits = model(input_ids=input_ids, use_cache=False).logits

        per_layer = []
        for idx, block in enumerate(model.transformer.h):
            def hook_fn(module, inp, out):
                if isinstance(out, tuple):
                    zeros = torch.zeros_like(out[0])
                    return (zeros,) + out[1:]
                return torch.zeros_like(out)

            handle = block.register_forward_hook(hook_fn)
            with torch.no_grad():
                ablated_logits = model(input_ids=input_ids, use_cache=False).logits
            handle.remove()

            drop = (base_logits - ablated_logits).abs().mean().item()
            per_layer.append(drop)

        all_per_layer.append(per_layer)

    return np.mean(all_per_layer, axis=0)


print("Computing logits ablation relevance for Model A (SciQ)...")
rel_A = compute_param_relevance_ablation("aiaa4051/task3/modelA/final", TEST_TEXTS_A)

print("Computing logits ablation relevance for Model B (SQuAD_v2)...")
rel_B = compute_param_relevance_ablation("aiaa4051/task3/modelB/final", TEST_TEXTS_B)

# Optional legacy proxy for comparison/debugging
print("Computing activation-magnitude proxy for Model A (SciQ)...")
proxy_A = compute_param_relevance_proxy("aiaa4051/task3/modelA/final", TEST_TEXTS_A)

print("Computing activation-magnitude proxy for Model B (SQuAD_v2)...")
proxy_B = compute_param_relevance_proxy("aiaa4051/task3/modelB/final", TEST_TEXTS_B)

# ── Normalize for fair comparison ─────────────────────────────────────────────
payload = build_param_relevance_payload(rel_A, rel_B, TEST_TEXTS_A, TEST_TEXTS_B)
payload["rel_A_proxy"] = proxy_A
payload["rel_B_proxy"] = proxy_B
payload["metric"] = "logits_ablation_mean_abs_drop"
rel_A_norm = payload["rel_A_norm"]
rel_B_norm = payload["rel_B_norm"]

with open("aiaa4051/task3/comparison/param_relevance_data.pkl", "wb") as f:
    pickle.dump(payload, f)
print("Saved raw parameter relevance data to aiaa4051/task3/comparison/param_relevance_data.pkl")

# ── Bar chart comparison ───────────────────────────────────────────────────────
layers = np.arange(len(rel_A))
width  = 0.35

fig, axes = plt.subplots(2, 1, figsize=(13, 8))

ax = axes[0]
ax.bar(layers - width/2, rel_A, width, label="Model A (SciQ)",     color="steelblue", alpha=0.85)
ax.bar(layers + width/2, rel_B, width, label="Model B (SQuAD_v2)", color="coral",     alpha=0.85)
ax.set_xlabel("Transformer Layer")
ax.set_ylabel("Parameter Relevance (abs)")
ax.set_title("Per-Layer MLP Parameter Relevance: Model A vs Model B")
ax.set_xticks(layers)
ax.legend()
ax.grid(axis="y", alpha=0.3)

ax = axes[1]
ax.bar(layers - width/2, rel_A_norm, width, label="Model A (SciQ)",     color="steelblue", alpha=0.85)
ax.bar(layers + width/2, rel_B_norm, width, label="Model B (SQuAD_v2)", color="coral",     alpha=0.85)
ax.set_xlabel("Transformer Layer")
ax.set_ylabel("Normalized relevance (fraction of total)")
ax.set_title("Normalized Per-Layer Relevance (relative contribution)")
ax.set_xticks(layers)
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("aiaa4051/task3/comparison/param_relevance_comparison.png", dpi=150)
print("Saved: aiaa4051/task3/comparison/param_relevance_comparison.png")

# ── Difference plot ────────────────────────────────────────────────────────────
diff = payload["diff_norm"]
top3 = np.argsort(np.abs(diff))[::-1][:3]

fig, ax = plt.subplots(figsize=(13, 3.5))
colors = ["#1a9850" if v > 0 else "#d73027" for v in diff]
ax.bar(layers, diff, color=colors, alpha=0.85)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Transformer Layer")
ax.set_ylabel("Δ Relevance (A − B)")
ax.set_title("Relevance Difference: Model A (SciQ) minus Model B (SQuAD_v2)\n"
             "Green = A relies more  |  Red = B relies more")
ax.set_xticks(layers)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("aiaa4051/task3/comparison/param_relevance_diff.png", dpi=150)
print("Saved: aiaa4051/task3/comparison/param_relevance_diff.png")

print(f"\nTop 3 layers with largest difference: {top3.tolist()}")
for l in top3:
    who = "Model A (SciQ)" if diff[l] > 0 else "Model B (SQuAD_v2)"
    print(f"  Layer {l:2d}: delta={diff[l]:+.4f} -> {who} relies more")
