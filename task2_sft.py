"""
Task 2-A: Sequential Fine-Tuning (SFT)
Step 1: Fine-tune GPT-2 on SQuAD_v2 (5000 samples, 2 epochs)
Step 2: Continue fine-tuning on SciQ   (3000 samples, 2 epochs)
Saves final checkpoint to aiaa4051/checkpoints/sft_final

Must run on GPU. Expected time: 4-8 h on RTX 4090.
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer
from datasets import load_from_disk

tokenizer = GPT2Tokenizer.from_pretrained("/home/user/project/gpt2")
tokenizer.pad_token = tokenizer.eos_token

# ── Tokenisation helpers ──────────────────────────────────────────────────────

def tokenize_squad(example):
    ans = example["answers"]["text"]
    answer_str = ans[0] if ans else "unanswerable"
    text = (f"Context: {example['context'][:300]}\n"
            f"Question: {example['question']}\n"
            f"Answer: {answer_str}")
    enc = tokenizer(text, truncation=True, max_length=256, padding="max_length")
    enc["labels"] = enc["input_ids"].copy()
    return enc

def tokenize_sciq(example):
    text = f"Question: {example['question']}\nAnswer: {example['correct_answer']}"
    enc = tokenizer(text, truncation=True, max_length=128, padding="max_length")
    enc["labels"] = enc["input_ids"].copy()
    return enc

# ── Step 1: SQuAD_v2 ─────────────────────────────────────────────────────────
print("=== Step 1: Fine-tuning on SQuAD_v2 ===")
model = GPT2LMHeadModel.from_pretrained("/home/user/project/gpt2")

squad = (load_from_disk("aiaa4051/data/squad_v2_train5000")
         .map(tokenize_squad, batched=False,
              remove_columns=["id","title","context","question","answers"]))
squad.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

args1 = TrainingArguments(
    output_dir="aiaa4051/checkpoints/step1_squad",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    save_strategy="epoch",
    logging_steps=100,
    fp16=True,
    report_to="none",
)
Trainer(model=model, args=args1, train_dataset=squad).train()
model.save_pretrained("aiaa4051/checkpoints/step1_squad/final")
print("Step 1 done → aiaa4051/checkpoints/step1_squad/final")

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
    per_device_train_batch_size=8,
    save_strategy="epoch",
    logging_steps=100,
    fp16=True,
    report_to="none",
)
Trainer(model=model, args=args2, train_dataset=sciq).train()
model.save_pretrained("aiaa4051/checkpoints/sft_final")
tokenizer.save_pretrained("aiaa4051/checkpoints/sft_final")
print("Sequential Fine-Tuning done → aiaa4051/checkpoints/sft_final")
