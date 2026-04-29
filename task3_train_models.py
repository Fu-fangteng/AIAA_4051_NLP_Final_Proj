"""
Task 3-A: Train Model A (SciQ) and Model B (SQuAD_v2) independently.
Must run on GPU. Expected time: ~2-4 h per model (RTX 4090).

Tip: run together with task2_sft.py in one GPU session to save time.
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer
from datasets import load_from_disk

tokenizer = GPT2Tokenizer.from_pretrained("/home/user/project/gpt2")
tokenizer.pad_token = tokenizer.eos_token

# ── Tokenisation helpers ──────────────────────────────────────────────────────

def tok_sciq(example):
    text = f"Question: {example['question']}\nAnswer: {example['correct_answer']}"
    enc  = tokenizer(text, truncation=True, max_length=128, padding="max_length")
    enc["labels"] = enc["input_ids"].copy()
    return enc

def tok_squad(example):
    ans = example["answers"]["text"]
    answer_str = ans[0] if ans else "unanswerable"
    text = (f"Context: {example['context'][:300]}\n"
            f"Question: {example['question']}\n"
            f"Answer: {answer_str}")
    enc = tokenizer(text, truncation=True, max_length=256, padding="max_length")
    enc["labels"] = enc["input_ids"].copy()
    return enc

# ── Generic trainer ───────────────────────────────────────────────────────────

def train_model(dataset_name, output_dir, tokenize_fn, n_samples=3000,
                remove_cols=None):
    print(f"\n=== Training on {dataset_name} → {output_dir} ===")
    model = GPT2LMHeadModel.from_pretrained("/home/user/project/gpt2")
    ds = load_from_disk(dataset_name)
    if remove_cols:
        ds = ds.map(tokenize_fn, batched=False, remove_columns=remove_cols)
    else:
        ds = ds.map(tokenize_fn, batched=False)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        fp16=True,
        logging_steps=100,
        save_strategy="epoch",
        report_to="none",
    )
    Trainer(model=model, args=args, train_dataset=ds).train()
    model.save_pretrained(output_dir + "/final")
    tokenizer.save_pretrained(output_dir + "/final")
    print(f"Saved → {output_dir}/final")

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
