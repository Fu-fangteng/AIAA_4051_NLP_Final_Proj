"""
Task 1-B: AttnLRP Inference + Relevance Computation
Compute token-level relevance scores using AttnLRP on GPT-2.
- Local debug: N_SAMPLES=5, ~5 min
- GPU full run: N_SAMPLES=200, ~1 h

Uses CP-LRP variant (stops gradient at Q and K in softmax).
Manual attention patch for compatibility with transformers 4.x.
"""

import torch
import pickle
from transformers import GPT2Tokenizer
from transformers.models.gpt2 import modeling_gpt2
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2MLP, GPT2Attention
from torch.nn import Dropout
from datasets import load_from_disk

from lxt.efficient.patches import patch_method, layer_norm_forward, dropout_forward
from lxt.efficient.models.gpt2 import mlp_forward
from lxt.efficient.rules import stop_gradient

# ── Patch MLP, LayerNorm, Dropout (compatible with transformers 4.x) ──
patch_method(mlp_forward, GPT2MLP)
patch_method(layer_norm_forward, modeling_gpt2.nn.LayerNorm)
patch_method(dropout_forward, Dropout)

# ── CP-LRP attention patch for GPT-2 / transformers 4.x ──
# CP-LRP stops gradient at Q and K (detaches softmax inputs),
# which is recommended for GPT-2 as it has negative logit values.
_original_gpt2_attn = GPT2Attention._attn.__func__ if hasattr(GPT2Attention._attn, '__func__') else GPT2Attention._attn

def _cp_lrp_attn(self, query, key, value, attention_mask=None, head_mask=None):
    query = stop_gradient(query)
    key   = stop_gradient(key)
    return _original_gpt2_attn(self, query, key, value, attention_mask, head_mask)

GPT2Attention._attn = _cp_lrp_attn
print("Patched: GPT2MLP, LayerNorm, Dropout, GPT2Attention (CP-LRP)")

# ── Set N_SAMPLES=5 for local debug; change to 200 before GPU cluster ──
N_SAMPLES = 200  # <-- change to 200 on cluster

tokenizer = GPT2Tokenizer.from_pretrained("/home/user/project/gpt2")
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = GPT2LMHeadModel.from_pretrained("/home/user/project/gpt2").to(device)
model.eval()
for param in model.parameters():
    param.requires_grad = False

dataset = load_from_disk("aiaa4051/data/squad_v2_dev200")
all_relevances = []

for i, sample in enumerate(dataset.select(range(N_SAMPLES))):
    input_ids = torch.tensor([sample["input_ids"]]).to(device)  # [1, seq_len]
    tokens = tokenizer.convert_ids_to_tokens(sample["input_ids"])

    # Input embeddings need gradient for relevance computation
    input_embeds = model.get_input_embeddings()(input_ids)
    input_embeds = input_embeds.detach().requires_grad_(True)

    # Forward pass
    logits = model(inputs_embeds=input_embeds, use_cache=False).logits

    # Backward from max logit at last position
    max_logit = logits[0, -1, :].max()
    max_logit.backward()

    # CP-LRP relevance: Input * Gradient, summed over embedding dim
    relevance = (input_embeds * input_embeds.grad).float().sum(-1).detach().cpu()[0].numpy()

    all_relevances.append({
        "tokens": tokens,
        "relevance": relevance,
        "input_ids": sample["input_ids"],
        "text": sample["text"],
    })

    if i % 10 == 0 or i == N_SAMPLES - 1:
        print(f"Processed {i+1}/{N_SAMPLES}")

with open("aiaa4051/task1/relevance/relevances.pkl", "wb") as f:
    pickle.dump(all_relevances, f)

print(f"\nDone. Saved {len(all_relevances)} records to aiaa4051/task1/relevance/relevances.pkl")
