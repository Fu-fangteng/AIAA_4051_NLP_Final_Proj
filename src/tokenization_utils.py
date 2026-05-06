"""Shared tokenization helpers for all training scripts."""


def apply_causal_lm_labels(enc, prompt_len):
    """Mask prompt tokens and padding from loss. Mutates enc in-place and returns it."""
    labels = enc["input_ids"].copy()
    labels[:prompt_len] = [-100] * prompt_len
    for i, mask in enumerate(enc["attention_mask"]):
        if mask == 0:
            labels[i] = -100
    enc["labels"] = labels
    return enc


def make_squad_tokenizer(tokenizer):
    """Return a tokenize function for SQuAD_v2 examples."""
    def tokenize(example):
        ans = example["answers"]["text"]
        answer_str = ans[0] if ans else "unanswerable"
        prompt = (
            f"Context: {example['context'][:300]}\n"
            f"Question: {example['question']}\n"
            f"Answer:"
        )
        full_text = prompt + f" {answer_str}"
        enc = tokenizer(full_text, truncation=True, max_length=256, padding="max_length")
        prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        return apply_causal_lm_labels(enc, prompt_len)
    return tokenize


def make_sciq_tokenizer(tokenizer):
    """Return a tokenize function for SciQ examples."""
    def tokenize(example):
        prompt = f"Question: {example['question']}\nAnswer:"
        full_text = prompt + f" {example['correct_answer']}"
        enc = tokenizer(full_text, truncation=True, max_length=128, padding="max_length")
        prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        return apply_causal_lm_labels(enc, prompt_len)
    return tokenize
