"""
Task 3-A: Train Model A (SciQ) and Model B (SQuAD_v2) independently.
Must run on GPU. Expected time: ~2-4 h per model (RTX 4090).

Tip: run together with task2_sft.py in one GPU session to save time.
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer
from datasets import load_from_disk

from training_config import (
    apply_model_memory_settings,
    base_model_path,
    model_dir_is_loadable,
    training_knobs,
)


BASE_MODEL = base_model_path()
TRAINING_KNOBS = training_knobs()

tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# ── Tokenisation helpers ──────────────────────────────────────────────────────

def tok_sciq(example):
    prompt    = f"Question: {example['question']}\nAnswer:"
    full_text = prompt + f" {example['correct_answer']}"
    enc = tokenizer(full_text, truncation=True, max_length=128, padding="max_length")

    prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    labels = enc["input_ids"].copy()
    labels[:prompt_len] = [-100] * prompt_len
    for i, mask in enumerate(enc["attention_mask"]):
        if mask == 0:
            labels[i] = -100
    enc["labels"] = labels
    return enc


def tok_squad(example):
    ans = example["answers"]["text"]
    answer_str = ans[0] if ans else "unanswerable"
    prompt = (f"Context: {example['context'][:300]}\n"
              f"Question: {example['question']}\n"
              f"Answer:")
    full_text = prompt + f" {answer_str}"
    enc = tokenizer(full_text, truncation=True, max_length=256, padding="max_length")

    prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    labels = enc["input_ids"].copy()
    labels[:prompt_len] = [-100] * prompt_len
    for i, mask in enumerate(enc["attention_mask"]):
        if mask == 0:
            labels[i] = -100
    enc["labels"] = labels
    return enc

# ── Generic trainer ───────────────────────────────────────────────────────────

def train_model(dataset_path, output_dir, tokenize_fn, n_samples,
                remove_cols=None):
    final_dir = output_dir + "/final"
    if model_dir_is_loadable(final_dir, GPT2LMHeadModel):
        print(f"\n=== Final checkpoint exists and loads; skipping: {final_dir} ===")
        return

    print(f"\n=== Training on {dataset_path} -> {output_dir} ===")
    model = GPT2LMHeadModel.from_pretrained(BASE_MODEL)
    apply_model_memory_settings(model)
    ds = load_from_disk(dataset_path).select(range(n_samples))
    if remove_cols:
        ds = ds.map(tokenize_fn, batched=False, remove_columns=remove_cols)
    else:
        ds = ds.map(tokenize_fn, batched=False)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=TRAINING_KNOBS["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAINING_KNOBS["gradient_accumulation_steps"],
        max_steps=TRAINING_KNOBS["max_steps"],
        fp16=True,
        logging_steps=100,
        save_strategy="epoch",
        save_total_limit=1,
        gradient_checkpointing=TRAINING_KNOBS["gradient_checkpointing"],
        report_to="none",
    )
    Trainer(model=model, args=args, train_dataset=ds).train()
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved: {final_dir}")

# ── Model A: SciQ ─────────────────────────────────────────────────────────────
train_model(
    "aiaa4051/data/sciq_train3000",
    "aiaa4051/task3/modelA",
    tok_sciq,
    n_samples=3000,
    remove_cols=["question", "distractor1", "distractor2",
                 "distractor3", "correct_answer", "support"],
)

# ── Model B: SQuAD_v2 ────────────────────────────────────────────────────────
train_model(
    "aiaa4051/data/squad_v2_train3000",
    "aiaa4051/task3/modelB",
    tok_squad,
    n_samples=3000,
    remove_cols=["id", "title", "context", "question", "answers"],
)
