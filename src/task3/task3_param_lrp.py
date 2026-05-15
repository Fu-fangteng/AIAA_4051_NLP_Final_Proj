"""
Task 3-B: Per-Layer MLP Parameter Relevance via AttnLRP

Computes parameter-level relevance using AttnLRP allocation rules:

    R_layer = Σ_{i,j} |W_ij × ∂logit/∂W_ij|

summed over c_fc and c_proj weight matrices for each transformer block,
then averaged over all test inputs.

Requires:
    - aiaa4051/task3/modelA/final   (from task3_train_models.py)
    - aiaa4051/task3/modelB/final   (from task3_train_models.py)

Output: aiaa4051/task3/comparison/param_relevance_data.pkl
Run task3_visualization.py afterwards to render the figure.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle
import numpy as np
import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from lxt_patch import apply_gpt2_cplrp
from task3_visualization import build_param_relevance_payload
from training_config import base_model_path

# Must be called before any model is loaded
apply_gpt2_cplrp(verbose=True)

tokenizer = GPT2Tokenizer.from_pretrained(base_model_path())
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

N_SAMPLES = 1000


def load_sciq_test_texts(n=N_SAMPLES):
    """Load SciQ test set, format as bare Question/Answer prompts."""
    ds = load_dataset("sciq", split="test")
    texts = []
    for item in ds:
        q = item["question"].strip()
        texts.append(f"Question: {q}\nAnswer:")
        if len(texts) >= n:
            break
    print(f"Loaded {len(texts)} SciQ test inputs")
    return texts


def load_squad_val_texts(n=N_SAMPLES):
    """Load SQuAD_v2 validation set (first n), format as Context/Question/Answer prompts."""
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    texts = []
    for item in ds:
        ctx = item["context"][:300].strip()
        q = item["question"].strip()
        texts.append(f"Context: {ctx}\nQuestion: {q}\nAnswer:")
        if len(texts) >= n:
            break
    print(f"Loaded {len(texts)} SQuAD_v2 validation inputs")
    return texts


def _load_model(model_path):
    """Load GPT-2 with CP-LRP patches and MLP weight gradients enabled."""
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    model.eval()
    for name, p in model.named_parameters():
        p.requires_grad = "mlp" in name
    return model


def compute_param_relevance_attnlrp(model_path, input_texts):
    """
    AttnLRP parameter relevance: R_layer = Σ |W × ∂logit/∂W|
    averaged over all input_texts.
    Covers both c_fc (768→3072) and c_proj (3072→768) per block.
    """
    model = _load_model(model_path)
    all_per_layer = []
    total = len(input_texts)

    for idx, input_text in enumerate(input_texts):
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"  [{idx+1}/{total}]")
        model.zero_grad()
        input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"].to(device)

        logits = model(input_ids=input_ids, use_cache=False).logits
        logits[0, -1, :].max().backward()

        per_layer = []
        for block in model.transformer.h:
            r = 0.0
            for W in [block.mlp.c_fc.weight, block.mlp.c_proj.weight]:
                if W.grad is not None:
                    r += (W * W.grad).abs().sum().item()
            per_layer.append(r)

        all_per_layer.append(per_layer)

    return np.mean(all_per_layer, axis=0)


print("Loading datasets...")
test_texts_A = load_sciq_test_texts(N_SAMPLES)
test_texts_B = load_squad_val_texts(N_SAMPLES)

print("\nComputing AttnLRP parameter relevance for Model A (SciQ)...")
rel_A = compute_param_relevance_attnlrp("aiaa4051/task3/modelA/final", test_texts_A)

print("\nComputing AttnLRP parameter relevance for Model B (SQuAD_v2)...")
rel_B = compute_param_relevance_attnlrp("aiaa4051/task3/modelB/final", test_texts_B)

payload = build_param_relevance_payload(rel_A, rel_B, test_texts_A, test_texts_B)
payload["metric"] = "attnlrp_weight_x_grad"
payload["n_samples"] = N_SAMPLES

out_path = Path("aiaa4051/task3/comparison/param_relevance_data.pkl")
with open(out_path, "wb") as f:
    pickle.dump(payload, f)
print(f"\nSaved: {out_path}")

diff = payload["diff_norm"]
top3 = np.argsort(np.abs(diff))[::-1][:3]
print(f"\nTop 3 layers with largest difference (N={N_SAMPLES} inputs each):")
for l in top3:
    who = "Model A (SciQ)" if diff[l] > 0 else "Model B (SQuAD_v2)"
    print(f"  Layer {int(l):2d}: delta={diff[l]:+.4f} → {who} relies more")

print(f"\nAll layer deltas (A-B):")
for i, d in enumerate(diff):
    print(f"  Layer {i:2d}: {d:+.4f}")
