"""
lxt_patch.py — Shared helper to apply CP-LRP patches to GPT-2.
Compatible with transformers 4.44.x.

Usage:
    from lxt_patch import apply_gpt2_cplrp, get_relevance
"""

import torch
from transformers.models.gpt2 import modeling_gpt2
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2MLP, GPT2Attention
from torch.nn import Dropout

from lxt.efficient.patches import patch_method, layer_norm_forward, dropout_forward
from lxt.efficient.models.gpt2 import mlp_forward
from lxt.efficient.rules import stop_gradient

_patched = False


def apply_gpt2_cplrp(verbose=False):
    """
    Patch GPT-2 for CP-LRP (one-time, idempotent).
    Must be called BEFORE loading the model.
    """
    global _patched
    if _patched:
        return
    patch_method(mlp_forward, GPT2MLP)
    patch_method(layer_norm_forward, modeling_gpt2.nn.LayerNorm)
    patch_method(dropout_forward, Dropout)

    _orig_attn = GPT2Attention._attn

    def _cp_lrp_attn(self, q, k, v, attention_mask=None, head_mask=None):
        return _orig_attn(self, stop_gradient(q), stop_gradient(k), v,
                          attention_mask, head_mask)

    GPT2Attention._attn = _cp_lrp_attn
    _patched = True
    if verbose:
        print("CP-LRP patches applied: GPT2MLP, LayerNorm, Dropout, GPT2Attention")


def get_relevance(model, input_ids, device):
    """
    Compute per-token CP-LRP relevance scores for a single sequence.

    Parameters
    ----------
    model  : GPT2LMHeadModel (already patched, eval mode, params frozen)
    input_ids : list[int]
    device : torch.device

    Returns
    -------
    numpy array of shape [seq_len]
    """
    ids = torch.tensor([input_ids]).to(device)
    emb = model.get_input_embeddings()(ids).detach().requires_grad_(True)
    logits = model(inputs_embeds=emb, use_cache=False).logits
    logits[0, -1, :].max().backward()
    return (emb * emb.grad).float().sum(-1).detach().cpu()[0].numpy()


def load_lrp_model(model_path, device):
    """
    Load a GPT-2 model (pretrained or fine-tuned) ready for CP-LRP inference.
    apply_gpt2_cplrp() must have been called first.
    """
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model
