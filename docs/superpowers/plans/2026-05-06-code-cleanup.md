# Code Cleanup — Reduce Duplication & Dead Code

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 刪除死碼、合併重複的 tokenizer 邏輯，並統一常用 helper，讓 codebase 在維持所有功能正常的前提下縮小程式量。

**Architecture:** 新增 `tokenization_utils.py` 放共用 tokenizer 工廠函數；在現有 `training_config.py` 補兩個 helper（`get_device`、`load_tokenizer`）；刪除 `faithfulness_utils.py` 裡從未被呼叫的函數及其對應測試；修除 unused import 與 poster render 裡的行內重複樣板。

**Tech Stack:** Python 3, PyTorch, Transformers (HuggingFace), pytest, matplotlib

---

## 檔案異動總覽

| 操作 | 檔案 |
|------|------|
| 新增 | `tokenization_utils.py` |
| 修改 | `training_config.py` — 加 `get_device()` 、`load_tokenizer()` |
| 修改 | `task2_sft.py` — 改用 shared utils |
| 修改 | `task3_train_models.py` — 改用 shared utils |
| 修改 | `benchmark_train_speed.py` — 改用 shared utils、移除 global |
| 修改 | `task1_attnlrp.py` — 改用 `get_device()` |
| 修改 | `task2_compare.py` — 改用 `get_device()` |
| 修改 | `faithfulness_utils.py` — 刪 `build_cumulative_mask_inputs` |
| 修改 | `tests/test_faithfulness_utils.py` — 刪對應 2 個測試 case |
| 修改 | `task1_visualize.py` — 刪 unused `mpatches` import |
| 修改 | `task2_render_poster.py` — 抽出行內 `_set_style()` |
| 修改 | `task3_render_poster.py` — 讓 diverging/grouped 用現有 `_clean_axis()` |

---

### Task 1: 在 `training_config.py` 加入共用 helpers

**Files:**
- Modify: `training_config.py`

- [ ] **Step 1: 在檔案尾端加入 `get_device()` 與 `load_tokenizer()`**

```python
# 加在 training_config.py 最底部

def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer(model_path=None):
    from transformers import GPT2Tokenizer
    if model_path is None:
        model_path = base_model_path()
    tok = GPT2Tokenizer.from_pretrained(model_path)
    tok.pad_token = tok.eos_token
    return tok
```

- [ ] **Step 2: 驗證 import 正常**

```bash
cd C:/Users/0610r/cc/AIAA_4051_NLP_Final_Proj
python -c "from training_config import get_device, load_tokenizer; print('OK')"
```

預期輸出：`OK`

- [ ] **Step 3: 執行現有測試確保沒有破壞**

```bash
pytest tests/ -v
```

預期：全部 PASS

- [ ] **Step 4: Commit**

```bash
git add training_config.py
git commit -m "feat: add get_device() and load_tokenizer() helpers to training_config"
```

---

### Task 2: 建立 `tokenization_utils.py`

**Files:**
- Create: `tokenization_utils.py`

這裡抽出所有訓練腳本共用的 labels-masking 邏輯與 tokenizer 工廠函數。

- [ ] **Step 1: 建立檔案**

```python
# tokenization_utils.py
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
```

- [ ] **Step 2: 驗證 import**

```bash
python -c "from tokenization_utils import apply_causal_lm_labels, make_squad_tokenizer, make_sciq_tokenizer; print('OK')"
```

預期：`OK`

- [ ] **Step 3: Commit**

```bash
git add tokenization_utils.py
git commit -m "feat: add tokenization_utils with shared label-masking and tokenizer factories"
```

---

### Task 3: 更新 `task2_sft.py`

**Files:**
- Modify: `task2_sft.py`

- [ ] **Step 1: 替換 import 區段與 tokenizer 初始化**

把原本的：
```python
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
```

改成：
```python
from transformers import GPT2LMHeadModel, TrainingArguments, Trainer
from datasets import load_from_disk

from training_config import (
    apply_model_memory_settings,
    base_model_path,
    load_tokenizer,
    model_dir_is_loadable,
    training_knobs,
)
from tokenization_utils import make_sciq_tokenizer, make_squad_tokenizer


BASE_MODEL = base_model_path()
TRAINING_KNOBS = training_knobs()
STEP1_FINAL = "aiaa4051/checkpoints/step1_squad/final"
SFT_FINAL = "aiaa4051/checkpoints/sft_final"

tokenizer = load_tokenizer(BASE_MODEL)
tokenize_squad = make_squad_tokenizer(tokenizer)
tokenize_sciq = make_sciq_tokenizer(tokenizer)
```

- [ ] **Step 2: 驗證 import**

```bash
python -c "
import sys; sys.path.insert(0, '.')
# Only test imports, don't execute training code
import ast, tokenize as tk
with open('task2_sft.py') as f:
    src = f.read()
# Quick syntax check
ast.parse(src)
print('Syntax OK')
"
```

預期：`Syntax OK`

- [ ] **Step 3: Commit**

```bash
git add task2_sft.py
git commit -m "refactor: task2_sft uses shared tokenization_utils and load_tokenizer"
```

---

### Task 4: 更新 `task3_train_models.py`

**Files:**
- Modify: `task3_train_models.py`

- [ ] **Step 1: 替換 import 區段與 tokenizer 初始化**

把原本的：
```python
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
```

改成：
```python
from transformers import GPT2LMHeadModel, TrainingArguments, Trainer
from datasets import load_from_disk

from training_config import (
    apply_model_memory_settings,
    base_model_path,
    load_tokenizer,
    model_dir_is_loadable,
    training_knobs,
)
from tokenization_utils import make_sciq_tokenizer, make_squad_tokenizer


BASE_MODEL = base_model_path()
TRAINING_KNOBS = training_knobs()

tokenizer = load_tokenizer(BASE_MODEL)
tok_sciq = make_sciq_tokenizer(tokenizer)
tok_squad = make_squad_tokenizer(tokenizer)
```

- [ ] **Step 2: 驗證語法**

```bash
python -c "import ast; ast.parse(open('task3_train_models.py').read()); print('Syntax OK')"
```

預期：`Syntax OK`

- [ ] **Step 3: Commit**

```bash
git add task3_train_models.py
git commit -m "refactor: task3_train_models uses shared tokenization_utils and load_tokenizer"
```

---

### Task 5: 更新 `benchmark_train_speed.py`

**Files:**
- Modify: `benchmark_train_speed.py`

同時修復 `tokenizer = None` 全域變數的 awkward 模式，改成在 `main()` 內傳遞。

- [ ] **Step 1: 整個檔案改寫如下**

```python
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
```

- [ ] **Step 2: 驗證語法**

```bash
python -c "import ast; ast.parse(open('benchmark_train_speed.py').read()); print('Syntax OK')"
```

預期：`Syntax OK`

- [ ] **Step 3: Commit**

```bash
git add benchmark_train_speed.py
git commit -m "refactor: benchmark uses shared utils, remove tokenizer global"
```

---

### Task 6: 在 `task1_attnlrp.py` 和 `task2_compare.py` 用 `get_device()`

**Files:**
- Modify: `task1_attnlrp.py`
- Modify: `task2_compare.py`

- [ ] **Step 1: 修改 `task1_attnlrp.py`**

把：
```python
from training_config import base_model_path
```
改成：
```python
from training_config import base_model_path, get_device
```

把：
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```
改成：
```python
device = get_device()
print(f"Using device: {device}")
```

- [ ] **Step 2: 修改 `task2_compare.py`**

把：
```python
from training_config import base_model_path
```
改成：
```python
from training_config import base_model_path, get_device
```

把：
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```
改成：
```python
device = get_device()
print(f"Using device: {device}")
```

- [ ] **Step 3: 驗證語法**

```bash
python -c "import ast; ast.parse(open('task1_attnlrp.py').read()); print('task1 OK')"
python -c "import ast; ast.parse(open('task2_compare.py').read()); print('task2 OK')"
```

預期兩行都輸出 `OK`

- [ ] **Step 4: Commit**

```bash
git add task1_attnlrp.py task2_compare.py
git commit -m "refactor: replace inline device selection with get_device()"
```

---

### Task 7: 刪除 `build_cumulative_mask_inputs` 及其測試

**Files:**
- Modify: `faithfulness_utils.py`
- Modify: `tests/test_faithfulness_utils.py`

- [ ] **Step 1: 從 `faithfulness_utils.py` 刪除整個函數（第 4-13 行）**

刪除以下內容：
```python
def build_cumulative_mask_inputs(tokens, sorted_idx, mask_token_id):
    if len(sorted_idx) != len(tokens):
        raise ValueError("sorted_idx must contain exactly one index per token")

    masked_tokens = list(tokens)
    masked_inputs = [masked_tokens.copy()]
    for idx in sorted_idx:
        masked_tokens[idx] = mask_token_id
        masked_inputs.append(masked_tokens.copy())
    return masked_inputs
```

同時，`tests/test_faithfulness_utils.py` 的 import 也要更新，把：
```python
from faithfulness_utils import (
    build_cumulative_attention_masks,
    build_cumulative_mask_inputs,
    least_relevant_first_order,
    mask_percent_axis,
    relevance_first_order,
)
```
改成：
```python
from faithfulness_utils import (
    build_cumulative_attention_masks,
    least_relevant_first_order,
    mask_percent_axis,
    relevance_first_order,
)
```

- [ ] **Step 2: 從測試檔刪除 2 個對應的 test methods**

刪除整個 `test_cumulative_mask_inputs_start_with_unmasked_tokens` 方法（lines 13-20）：
```python
    def test_cumulative_mask_inputs_start_with_unmasked_tokens(self):
        tokens = [10, 20, 30]
        masked = build_cumulative_mask_inputs(tokens, [2, 0, 1], mask_token_id=99)

        self.assertEqual(masked[0], [10, 20, 30])
        self.assertEqual(masked[1], [10, 20, 99])
        self.assertEqual(masked[2], [99, 20, 99])
        self.assertEqual(masked[3], [99, 99, 99])
```

刪除整個 `test_cumulative_mask_inputs_validate_order_length` 方法（lines 22-24）：
```python
    def test_cumulative_mask_inputs_validate_order_length(self):
        with self.assertRaises(ValueError):
            build_cumulative_mask_inputs([10, 20, 30], [0, 1], mask_token_id=99)
```

- [ ] **Step 3: 執行測試確認剩下的測試還在 PASS**

```bash
pytest tests/test_faithfulness_utils.py -v
```

預期：5 個測試全部 PASS（`build_cumulative_attention_masks` 相關的 2 個 + `mask_percent_axis` + `relevance_first_order` + `least_relevant_first_order`）

- [ ] **Step 4: Commit**

```bash
git add faithfulness_utils.py tests/test_faithfulness_utils.py
git commit -m "chore: delete unused build_cumulative_mask_inputs and its tests"
```

---

### Task 8: 修理 poster render 裡的重複樣板

**Files:**
- Modify: `task2_render_poster.py` — 抽出行內 style 設定為 `_set_style()`
- Modify: `task3_render_poster.py` — 讓 `render_comparison_diverging()` 和 `render_comparison_grouped()` 使用既有的 `_clean_axis()`

- [ ] **Step 1: 修改 `task2_render_poster.py`**

在 `render_question_heatmap()` 函數**之前**插入新函數：
```python
def _set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#E5E7EB",
        "axes.labelcolor": "#111827",
        "xtick.color": "#111827",
        "ytick.color": "#111827",
    })
```

然後在 `render_question_heatmap()` 函數開頭，把原本行內的 `plt.rcParams.update({...})` 那一整段（原本 lines 75-83）替換成：
```python
    _set_style()
```

- [ ] **Step 2: 修改 `task3_render_poster.py` 的 `render_comparison_diverging()`**

把函數裡面行內的這段（在 `fig.tight_layout` 之前）：
```python
    # Clean axis
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.spines["bottom"].set_linewidth(0.9)
```
替換成：
```python
    _clean_axis(ax)
```

- [ ] **Step 3: 修改 `task3_render_poster.py` 的 `render_comparison_grouped()`**

同樣，把函數裡面行內的這段：
```python
    # Clean axis
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.spines["bottom"].set_linewidth(0.9)
```
替換成：
```python
    _clean_axis(ax)
```

- [ ] **Step 4: 驗證語法**

```bash
python -c "import ast; ast.parse(open('task2_render_poster.py').read()); print('task2_render OK')"
python -c "import ast; ast.parse(open('task3_render_poster.py').read()); print('task3_render OK')"
```

預期兩行都輸出 `OK`

- [ ] **Step 5: Commit**

```bash
git add task2_render_poster.py task3_render_poster.py
git commit -m "refactor: extract _set_style() in task2_render_poster, use _clean_axis() in task3_render_poster"
```

---

### Task 9: 移除未使用的 import

**Files:**
- Modify: `task1_visualize.py`

- [ ] **Step 1: 刪除 `task1_visualize.py` 第 9 行的 unused import**

刪除這一行：
```python
import matplotlib.patches as mpatches
```

- [ ] **Step 2: 驗證語法**

```bash
python -c "import ast; ast.parse(open('task1_visualize.py').read()); print('Syntax OK')"
```

預期：`Syntax OK`

- [ ] **Step 3: Commit**

```bash
git add task1_visualize.py
git commit -m "chore: remove unused matplotlib.patches import from task1_visualize"
```

---

### Task 10: 最終驗證

**Files:** — (no changes)

- [ ] **Step 1: 執行完整測試套件**

```bash
pytest tests/ -v
```

預期：所有測試 PASS，無 FAIL 無 ERROR

- [ ] **Step 2: 驗證所有主要模組 import 正常**

```bash
python -c "
import ast
files = [
    'training_config.py',
    'tokenization_utils.py',
    'faithfulness_utils.py',
    'task1_attnlrp.py',
    'task1_visualize.py',
    'task2_compare.py',
    'task2_sft.py',
    'task2_render_poster.py',
    'task3_train_models.py',
    'task3_render_poster.py',
    'benchmark_train_speed.py',
]
for f in files:
    ast.parse(open(f).read())
    print(f'  {f}: OK')
print('All syntax checks passed.')
"
```

預期：每個檔案都印出 `OK`，最後印出 `All syntax checks passed.`

- [ ] **Step 3: 確認 `build_cumulative_mask_inputs` 已完全消失**

```bash
grep -r "build_cumulative_mask_inputs" . --include="*.py"
```

預期：**無輸出**（代表已完全移除）

- [ ] **Step 4: 確認沒有遺漏的 `torch.device("cuda"` 重複模式**

```bash
grep -n 'torch.device("cuda"' task1_attnlrp.py task2_compare.py task2_sft.py task3_train_models.py benchmark_train_speed.py 2>/dev/null || echo "None found - clean"
```

預期：`None found - clean`
