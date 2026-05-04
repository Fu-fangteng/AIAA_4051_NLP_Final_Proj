"""
Task 1-A: Data Preprocessing
Load SQuAD_v2 validation set (first 200 samples) and tokenize for GPT-2.
Run locally on Mac. Expected time: ~10 min.
"""

from datasets import load_dataset
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load SQuAD_v2, first 200 samples from validation set
dataset = load_dataset("squad_v2", split="validation[:200]")
print(f"Loaded {len(dataset)} samples.")

def preprocess(example):
    # Concatenate context + question as GPT-2 input
    text = f"Context: {example['context'][:300]}\nQuestion: {example['question']}\nAnswer:"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    return {
        "input_ids":      inputs["input_ids"][0].tolist(),
        "attention_mask": inputs["attention_mask"][0].tolist(),
        "text": text,
    }

processed = dataset.map(preprocess, remove_columns=dataset.column_names)
processed.save_to_disk("aiaa4051/data/squad_v2_dev200")
print(f"Saved {len(processed)} processed samples to aiaa4051/data/squad_v2_dev200")
