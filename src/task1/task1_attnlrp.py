"""
Task 1-B: AttnLRP Inference + Relevance Computation
Compute token-level relevance scores using AttnLRP on GPT-2.
- Local debug: N_SAMPLES=5, ~5 min
- GPU full run: N_SAMPLES=200, ~1 h

Uses CP-LRP variant (stops gradient at Q and K in softmax).
Patches are applied via the shared lxt_patch.py helper.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pickle
from transformers import GPT2Tokenizer
from datasets import load_from_disk

from lxt_patch import apply_gpt2_cplrp, get_relevance, load_lrp_model
from training_config import base_model_path, get_device

apply_gpt2_cplrp(verbose=True)

# ── Set N_SAMPLES=5 for local debug; change to 200 before GPU cluster ──
N_SAMPLES = 200  # <-- change to 200 on cluster

BASE_MODEL = base_model_path()

tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

device = get_device()
print(f"Using device: {device}")

model = load_lrp_model(BASE_MODEL, device)

dataset = load_from_disk("aiaa4051/data/squad_v2_dev200")
all_relevances = []

for i, sample in enumerate(dataset.select(range(N_SAMPLES))):
    tokens    = tokenizer.convert_ids_to_tokens(sample["input_ids"])
    relevance = get_relevance(model, sample["input_ids"], device)

    all_relevances.append({
        "tokens":    tokens,
        "relevance": relevance,
        "input_ids": sample["input_ids"],
        "text":      sample["text"],
    })

    if i % 10 == 0 or i == N_SAMPLES - 1:
        print(f"Processed {i+1}/{N_SAMPLES}")

with open("aiaa4051/task1/relevance/relevances.pkl", "wb") as f:
    pickle.dump(all_relevances, f)

print(f"\nDone. Saved {len(all_relevances)} records to aiaa4051/task1/relevance/relevances.pkl")
