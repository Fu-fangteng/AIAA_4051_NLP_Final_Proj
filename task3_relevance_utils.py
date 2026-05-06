import numpy as np


def build_param_relevance_payload(rel_a, rel_b, test_texts_a, test_texts_b):
    rel_a = np.asarray(rel_a, dtype=float)
    rel_b = np.asarray(rel_b, dtype=float)
    if rel_a.shape != rel_b.shape:
        raise ValueError("rel_a and rel_b must have the same shape")

    rel_a_norm = rel_a / (rel_a.sum() + 1e-8)
    rel_b_norm = rel_b / (rel_b.sum() + 1e-8)

    return {
        "rel_A": rel_a,
        "rel_B": rel_b,
        "rel_A_norm": rel_a_norm,
        "rel_B_norm": rel_b_norm,
        "diff_norm": rel_a_norm - rel_b_norm,
        "layers": np.arange(len(rel_a)),
        "test_texts_A": list(test_texts_a),
        "test_texts_B": list(test_texts_b),
        "model_A": "SciQ",
        "model_B": "SQuAD_v2",
        "metric": "activation_magnitude_proxy",
    }


def top_abs_difference_layers(diff, n=3):
    diff = np.asarray(diff, dtype=float)
    return np.argsort(np.abs(diff))[::-1][:n]
