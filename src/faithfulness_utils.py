import numpy as np


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
