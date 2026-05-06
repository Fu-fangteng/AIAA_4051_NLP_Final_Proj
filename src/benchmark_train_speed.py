import time

import torch
from datasets import load_from_disk
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments

from training_config import (
    apply_model_memory_settings,
    base_model_path,
    get_device,
    load_tokenizer,
    training_knobs,
)
from tokenization_utils import make_squad_tokenizer


def main():
    knobs = training_knobs()
    max_steps = knobs["max_steps"] if knobs["max_steps"] > 0 else 20
    model_path = base_model_path()
    device = get_device()

    print(f"Base model: {model_path}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = load_tokenizer(model_path)
    tokenize_squad = make_squad_tokenizer(tokenizer)

    model = GPT2LMHeadModel.from_pretrained(model_path)
    apply_model_memory_settings(model)

    ds = (
        load_from_disk("aiaa4051/data/squad_v2_train5000")
        .select(range(64))
        .map(
            tokenize_squad,
            batched=False,
            remove_columns=["id", "title", "context", "question", "answers"],
        )
    )
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    args = TrainingArguments(
        output_dir="aiaa4051/benchmark/train_speed",
        overwrite_output_dir=True,
        max_steps=max_steps,
        per_device_train_batch_size=knobs["per_device_train_batch_size"],
        gradient_accumulation_steps=knobs["gradient_accumulation_steps"],
        fp16=True,
        gradient_checkpointing=knobs["gradient_checkpointing"],
        save_strategy="no",
        logging_steps=1,
        report_to="none",
    )

    start = time.time()
    result = Trainer(model=model, args=args, train_dataset=ds).train()
    elapsed = time.time() - start
    print(f"Benchmark elapsed seconds: {elapsed:.2f}")
    print(f"Benchmark train result: {result.metrics}")
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Peak CUDA memory allocated GB: {peak_gb:.2f}")


if __name__ == "__main__":
    main()
