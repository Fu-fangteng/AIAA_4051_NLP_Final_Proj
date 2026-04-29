"""
Task 3-B: Per-Layer Parameter Relevance Comparison (Model A vs Model B)
Extends CP-LRP to the MLP weight level using LRP-epsilon rule:
    R_W ∝ |W * x|  summed over each Transformer layer's MLP.

Requires:
  - aiaa4051/task3/modelA/final   (from task3_train_models.py)
  - aiaa4051/task3/modelB/final   (from task3_train_models.py)
  - lxt_patch.py                  (shared CP-LRP helper)

Run on Mac or GPU. Expected time: 20 min (CPU) / 5 min (GPU).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer

from lxt_patch import apply_gpt2_cplrp, load_lrp_model

apply_gpt2_cplrp(verbose=True)

tokenizer = GPT2Tokenizer.from_pretrained("/home/user/project/gpt2")
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Per-layer parameter relevance ─────────────────────────────────────────────

def compute_param_relevance(model_path, input_text):
    """
    Returns a list of floats (one per Transformer layer) representing
    the total parameter relevance of each layer's MLP c_fc weights.
    Uses LRP-epsilon rule: R ∝ |W * mean_activation|.
    """
    model = load_lrp_model(model_path, device)
    tokens = tokenizer(input_text, return_tensors="pt")
    input_ids = tokens["input_ids"].to(device)

    # Register hooks to capture MLP input activations
    layer_inputs = {}
    hooks = []
    for idx, block in enumerate(model.transformer.h):
        def make_hook(i):
            def hook_fn(module, inp, out):
                layer_inputs[i] = inp[0].detach()
            return hook_fn
        hooks.append(block.mlp.c_fc.register_forward_hook(make_hook(idx)))

    # Forward + backward
    emb = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
    logits = model(inputs_embeds=emb, use_cache=False).logits
    logits[0, -1, :].max().backward()

    for h in hooks:
        h.remove()

    # LRP-epsilon: R_layer = sum |W * x_mean|
    param_relevances = []
    for idx, block in enumerate(model.transformer.h):
        W = block.mlp.c_fc.weight.detach()   # [out_features, in_features]
        if idx in layer_inputs:
            x = layer_inputs[idx]            # [1, seq_len, in_features]
            x_mean = x.mean(dim=1)           # [1, in_features]
            R = (x_mean @ W).abs().sum().item()
        else:
            R = 0.0
        param_relevances.append(R)

    return param_relevances

# ── Test inputs (use several to get a more stable picture) ───────────────────
test_texts = [
    "Context: Water is a chemical compound of hydrogen and oxygen.\nQuestion: What is water?\nAnswer:",
    "Context: The sun is a star at the center of the solar system.\nQuestion: What is the sun?\nAnswer:",
    "Context: Photosynthesis is the process by which plants make food.\nQuestion: How do plants make food?\nAnswer:",
]

print("Computing parameter relevance for Model A (SciQ)...")
rel_A_all = [compute_param_relevance("aiaa4051/task3/modelA/final", t) for t in test_texts]
rel_A = np.mean(rel_A_all, axis=0)

print("Computing parameter relevance for Model B (SQuAD_v2)...")
rel_B_all = [compute_param_relevance("aiaa4051/task3/modelB/final", t) for t in test_texts]
rel_B = np.mean(rel_B_all, axis=0)

# ── Normalize for fair comparison ─────────────────────────────────────────────
rel_A_norm = rel_A / (rel_A.sum() + 1e-8)
rel_B_norm = rel_B / (rel_B.sum() + 1e-8)

# ── Bar chart comparison ───────────────────────────────────────────────────────
layers = np.arange(len(rel_A))
width  = 0.35

fig, axes = plt.subplots(2, 1, figsize=(13, 8))

# Absolute values
ax = axes[0]
ax.bar(layers - width/2, rel_A, width, label="Model A (SciQ)",     color="steelblue", alpha=0.85)
ax.bar(layers + width/2, rel_B, width, label="Model B (SQuAD_v2)", color="coral",     alpha=0.85)
ax.set_xlabel("Transformer Layer")
ax.set_ylabel("Parameter Relevance (abs)")
ax.set_title("Per-Layer MLP Parameter Relevance: Model A vs Model B")
ax.set_xticks(layers)
ax.legend()
ax.grid(axis="y", alpha=0.3)

# Normalized (relative contribution)
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
diff = rel_A_norm - rel_B_norm
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
    print(f"  Layer {l:2d}: Δ={diff[l]:+.4f}  → {who} relies more")
