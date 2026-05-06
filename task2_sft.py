"""
Task 2-A: Sequential Fine-Tuning (SFT)
Step 1: Fine-tune GPT-2 on SQuAD_v2 (5000 samples, 2 epochs)
Step 2: Continue fine-tuning on SciQ   (3000 samples, 2 epochs)
Saves final checkpoint to aiaa4051/checkpoints/sft_final

Must run on GPU. Expected time: 4-8 h on RTX 4090.
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
STEP1_FINAL = "aiaa4051/checkpoints/step1_squad/final"
SFT_FINAL = "aiaa4051/checkpoints/sft_final"

tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# ── Tokenisation helpers ──────────────────────────────────────────────────────

def tokenize_squad(example):
    ans = example["answers"]["text"]
    answer_str = ans[0] if ans else "unanswerable"
    prompt = (f"Context: {example['context'][:300]}\n"
              f"Question: {example['question']}\n"
              f"Answer:")
    full_text = prompt + f" {answer_str}"
    enc = tokenizer(full_text, truncation=True, max_length=256, padding="max_length")

    # Mask prompt tokens from loss so the model only learns to predict the answer
    prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    labels = enc["input_ids"].copy()
    labels[:prompt_len] = [-100] * prompt_len
    for i, mask in enumerate(enc["attention_mask"]):
        if mask == 0:
            labels[i] = -100
    enc["labels"] = labels
    return enc


def tokenize_sciq(example):
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

# ── Step 1: SQuAD_v2 ─────────────────────────────────────────────────────────
print("=== Step 1: Fine-tuning on SQuAD_v2 ===")
if model_dir_is_loadable(STEP1_FINAL, GPT2LMHeadModel):
    print(f"Step 1 final exists and loads; skipping Step 1: {STEP1_FINAL}")
    model = GPT2LMHeadModel.from_pretrained(STEP1_FINAL)
    apply_model_memory_settings(model)
else:
    model = GPT2LMHeadModel.from_pretrained(BASE_MODEL)
    apply_model_memory_settings(model)

    squad = (load_from_disk("aiaa4051/data/squad_v2_train5000")
             .map(tokenize_squad, batched=False,
                  remove_columns=["id","title","context","question","answers"]))
    squad.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    args1 = TrainingArguments(
        output_dir="aiaa4051/checkpoints/step1_squad",
        num_train_epochs=2,
        per_device_train_batch_size=TRAINING_KNOBS["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAINING_KNOBS["gradient_accumulation_steps"],
        max_steps=TRAINING_KNOBS["max_steps"],
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=100,
        fp16=True,
        gradient_checkpointing=TRAINING_KNOBS["gradient_checkpointing"],
        report_to="none",
    )
    Trainer(model=model, args=args1, train_dataset=squad).train()
    model.save_pretrained(STEP1_FINAL)
    tokenizer.save_pretrained(STEP1_FINAL)
    print(f"Step 1 done: {STEP1_FINAL}")

# ── Step 2: SciQ ─────────────────────────────────────────────────────────────
print("\n=== Step 2: Continue fine-tuning on SciQ ===")
sciq = (load_from_disk("aiaa4051/data/sciq_train3000")
        .map(tokenize_sciq, batched=False,
             remove_columns=["question","distractor1","distractor2",
                             "distractor3","correct_answer","support"]))
sciq.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

args2 = TrainingArguments(
    output_dir="aiaa4051/checkpoints/step2_sciq",
    num_train_epochs=2,
    per_device_train_batch_size=TRAINING_KNOBS["per_device_train_batch_size"],
    gradient_accumulation_steps=TRAINING_KNOBS["gradient_accumulation_steps"],
    max_steps=TRAINING_KNOBS["max_steps"],
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=100,
    fp16=True,
    gradient_checkpointing=TRAINING_KNOBS["gradient_checkpointing"],
    report_to="none",
)
Trainer(model=model, args=args2, train_dataset=sciq).train()
model.save_pretrained(SFT_FINAL)
tokenizer.save_pretrained(SFT_FINAL)
print(f"Sequential Fine-Tuning done: {SFT_FINAL}")
