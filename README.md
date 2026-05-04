# AttnLRP on GPT-2 Small
**AIAA 4051 — NLP Final Project**

Applying CP-LRP (AttnLRP) to GPT-2 Small for transformer explainability across three levels of analysis: token attribution, fine-tuning effects, and parameter-level layer contributions.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Setup & Installation](#3-setup--installation)
4. [Running the Pipeline](#4-running-the-pipeline)
   - [Task 1 — Token Relevance](#task-1--token-relevance-50-pts)
   - [Task 2 — Fine-Tuning Comparison](#task-2--fine-tuning-comparison-30-pts)
   - [Task 3 — Parameter-Level Analysis](#task-3--parameter-level-analysis-20-pts)
5. [Results Summary](#5-results-summary)
6. [Strengths & Limitations](#6-strengths--limitations)
7. [Improvement Roadmap](#7-improvement-roadmap)
8. [References](#8-references)

---

## 1. Project Overview

This project implements and evaluates **AttnLRP** (Achtibat et al., ICML 2024), a Layer-wise Relevance Propagation (LRP) variant designed for transformer architectures. The specific variant used is **CP-LRP**, which stops gradient flow at the Query and Key tensors of the attention softmax — the recommended choice for GPT-2 due to its unbounded logit values.

All experiments use **GPT-2 Small** (124M parameters, 12 Transformer layers) as the base model.

### Three Tasks

| Task | Description | Points |
|---|---|---|
| **Task 1** | Reproduce AttnLRP on SQuAD_v2 (200 samples): compute signed token relevance, visualize heatmaps, evaluate faithfulness via Pixel Flipping | 50 |
| **Task 2** | Sequential Fine-Tuning (SQuAD_v2 → SciQ), compare AttnLRP attributions before and after on 20 samples | 30 |
| **Task 3** | Train Model A (SciQ) and Model B (SQuAD_v2) independently, compare per-layer MLP parameter relevance using LRP-ε rule | 20 |

### Key Technical Choices

- **CP-LRP rule**: `stop_gradient` applied to Q and K before the attention softmax; gradient flows freely through V and all MLP/LayerNorm layers.
- **Relevance formula**: $R_i = \sum_d e_{i,d} \cdot \frac{\partial \hat{y}}{\partial e_{i,d}}$ — element-wise product of input embedding and its gradient, summed over the embedding dimension.
- **LRP library**: [`lxt` (LRP-eXplains-Transformers)](https://github.com/rachtibat/LRP-eXplains-Transformers) with manual compatibility patches for `transformers 4.44.x`.
- **Parameter-level LRP (Task 3)**: approximate LRP-ε rule using mean sequence activation: $R_\ell = \sum_{i,j} |\bar{x}_j \cdot W_{ji}^{(c\_fc)}|$.

---

## 2. Repository Structure

```
project/
├── README.md                    ← this file
├── REPORT.md                    ← full written report with analysis and findings
├── requirements.txt             ← Python dependencies
│
├── lxt_patch.py                 ← shared CP-LRP patch helper (used by all scripts)
├── test.py                      ← environment / GPU connectivity test
│
├── task1_data_prep.py           ← Task 1-A: download and preprocess SQuAD_v2 (local)
├── task1_attnlrp.py             ← Task 1-B: compute CP-LRP relevance scores (GPU)
├── task1_visualize.py           ← Task 1-C: generate token relevance heatmaps (local)
├── task1_faithfulness.py        ← Task 1-D: Pixel Flipping faithfulness eval (GPU)
│
├── task2_sft.py                 ← Task 2-A: Sequential Fine-Tuning on SQuAD_v2→SciQ (GPU)
├── task2_compare.py             ← Task 2-B: pre/post-FT AttnLRP comparison (GPU/local)
│
├── task3_train_models.py        ← Task 3-A: train Model A (SciQ) + Model B (SQuAD) (GPU)
├── task3_param_lrp.py           ← Task 3-B: per-layer parameter relevance comparison (local/GPU)
│
├── gpt2/                        ← GPT-2 Small weights (local copy for cluster use)
│   ├── model.safetensors
│   ├── config.json
│   ├── vocab.json
│   └── merges.txt
│
├── LRP-eXplains-Transformers/   ← lxt library (git clone + pip install -e .)
│
└── aiaa4051/                    ← all outputs (auto-created by scripts)
    ├── data/
    │   ├── squad_v2_dev200/     ← 200-sample SQuAD_v2 validation set (Task 1 input)
    │   ├── squad_v2_train3000/  ← SQuAD_v2 training subset (Task 3 Model B)
    │   ├── squad_v2_train5000/  ← SQuAD_v2 training subset (Task 2 SFT step 1)
    │   └── sciq_train3000/      ← SciQ training subset (Task 2 step 2 + Task 3 Model A)
    │
    ├── task1/
    │   ├── relevance/
    │   │   ├── relevances.pkl   ← tokens + relevance scores for all 200 samples
    │   │   └── sample_0.png … sample_199.png
    │   └── faithfulness/
    │       ├── faithfulness_curve.png
    │       └── faithfulness_data.pkl
    │
    ├── task2/
    │   └── comparison/
    │       ├── comparison.pkl   ← pre/post relevance for 20 samples
    │       ├── compare_sample0.png … compare_sample4.png
    │       └── mean_delta.png
    │
    ├── task3/
    │   ├── modelA/              ← GPT-2 fine-tuned on SciQ (3 epochs)
    │   │   ├── checkpoint-375/
    │   │   ├── checkpoint-750/
    │   │   ├── checkpoint-1125/
    │   │   └── final/           ← model weights + tokenizer
    │   ├── modelB/              ← GPT-2 fine-tuned on SQuAD_v2 (3 epochs)
    │   │   └── final/
    │   └── comparison/
    │       ├── param_relevance_comparison.png
    │       └── param_relevance_diff.png
    │
    ├── checkpoints/
    │   ├── step1_squad/         ← SFT intermediate (after SQuAD_v2 step)
    │   │   └── final/
    │   ├── step2_sciq/          ← SFT intermediate checkpoints
    │   └── sft_final/           ← final SFT model (SQuAD→SciQ, used in Task 2)
    │
    └── logs/                    ← nohup output from cluster runs
        ├── task1_attnlrp.log
        ├── task1_faithfulness.log
        ├── task2_sft.log
        ├── task2_compare.log
        ├── task3_train_models.log
        └── task3_param_lrp.log
```

---

## 3. Setup & Installation

### Prerequisites

- Python 3.10+
- CUDA GPU recommended for training scripts (Tasks 2-A, 3-A) and large-scale inference (Tasks 1-B, 1-D)
- Mac/CPU sufficient for data prep and visualization scripts

### Step 1 — Create virtual environment

```bash
python3 -m venv attnlrp_env
source attnlrp_env/bin/activate
```

### Step 2 — Install Python dependencies

```bash
pip install torch torchvision transformers==4.44.2 datasets
pip install matplotlib seaborn numpy tqdm
```

> **Important**: `transformers 4.44.x` is required. The CP-LRP attention patch targets the internal `GPT2Attention._attn` method; newer versions may break the patch. See `lxt_patch.py` for details.

### Step 3 — Install the lxt library

```bash
cd LRP-eXplains-Transformers
pip install -e .
cd ..
```

### Step 4 — Prepare GPT-2 model weights

**Option A — Local weights (recommended for cluster with restricted internet):**
```bash
# Already present in gpt2/ — scripts load from /home/user/project/gpt2
# Change this path in each script if your directory differs
```

**Option B — Download from HuggingFace:**
```python
# In each script, replace:
#   GPT2LMHeadModel.from_pretrained("/home/user/project/gpt2")
# with:
#   GPT2LMHeadModel.from_pretrained("gpt2")
```

### Step 5 — Create output directories

```bash
mkdir -p aiaa4051/{data,logs}
mkdir -p aiaa4051/task1/{relevance,faithfulness}
mkdir -p aiaa4051/task2/comparison
mkdir -p aiaa4051/task3/{modelA,modelB,comparison}
mkdir -p aiaa4051/checkpoints
```

### Step 6 — Verify the environment

```bash
python test.py
```

Expected output: all 5 checks pass, confirming CUDA availability, model loading, `stop_gradient` patch, and gradient backpropagation.

---

## 4. Running the Pipeline

Scripts must be run **from the `project/` directory** (not from subdirectories), because all path references are relative to that root.

```bash
cd /path/to/project
```

---

### Task 1 — Token Relevance (50 pts)

#### 1-A: Data Preprocessing (local, ~10 min)

```bash
python task1_data_prep.py
```

Downloads SQuAD_v2 validation set, takes the first 200 samples, tokenizes with GPT-2 tokenizer (max length 256), and saves to `aiaa4051/data/squad_v2_dev200`.

**Output:** `aiaa4051/data/squad_v2_dev200/`

---

#### 1-B: AttnLRP Relevance Computation (GPU, ~1 h for N=200)

```bash
# On GPU cluster:
nohup python task1_attnlrp.py > aiaa4051/logs/task1_attnlrp.log 2>&1 &

# Local debug (edit N_SAMPLES = 5 at line 42 first):
python task1_attnlrp.py
```

Applies the CP-LRP patch via `lxt_patch.py`, then for each sample:
1. Computes input embeddings
2. Runs forward pass
3. Backpropagates from `max logit at final position`
4. Relevance = `(embedding * gradient).sum(dim=-1)` → signed scalar per token

**Output:** `aiaa4051/task1/relevance/relevances.pkl`

---

#### 1-C: Visualization (local, ~5 min)

```bash
python task1_visualize.py
```

Generates one PNG per sample: multi-row wrapped heatmap with `RdYlGn` colormap (green = promotes prediction, red = suppresses). Signed normalization centered at 0.

**Output:** `aiaa4051/task1/relevance/sample_0.png` … `sample_199.png`

---

#### 1-D: Faithfulness Evaluation (GPU, ~2–4 h)

```bash
nohup python task1_faithfulness.py > aiaa4051/logs/task1_faithfulness.log 2>&1 &
```

Runs Pixel Flipping on all 200 samples:
- **AttnLRP curve**: mask tokens in descending relevance order
- **Random baseline**: mask tokens in random order (seed=42)
- Plots both curves and reports normalized AUC

**Output:**
- `aiaa4051/task1/faithfulness/faithfulness_curve.png`
- `aiaa4051/task1/faithfulness/faithfulness_data.pkl`

**Current results:** AttnLRP AUC = 0.1359, Random AUC = 0.1245. The AttnLRP curve is non-monotonic: confidence initially drops as positive-relevance tokens are masked, then **rises** when masking crosses into negative-relevance (suppressive) tokens — whose removal releases inhibitory effects. See §6 Limitations for discussion.

---

### Task 2 — Fine-Tuning Comparison (30 pts)

#### 2-A: Sequential Fine-Tuning (GPU, ~4–8 h)

```bash
nohup python task2_sft.py > aiaa4051/logs/task2_sft.log 2>&1 &
```

**Data required** (create with `download_datasets.py` or HuggingFace, already saved to disk):
- `aiaa4051/data/squad_v2_train5000`
- `aiaa4051/data/sciq_train3000`

Training sequence:
1. Fine-tune GPT-2 on SQuAD_v2 (`train[:5000]`, 2 epochs, fp16, batch size 8) → `aiaa4051/checkpoints/step1_squad/final`
2. Continue fine-tuning the same model on SciQ (`train[:3000]`, 2 epochs) → `aiaa4051/checkpoints/sft_final`

**Output:** `aiaa4051/checkpoints/sft_final/` (model + tokenizer)

---

#### 2-B: Pre/Post Comparison (GPU or local, ~1–2 h)

```bash
nohup python task2_compare.py > aiaa4051/logs/task2_compare.log 2>&1 &
```

**Requires:** `aiaa4051/checkpoints/sft_final/` (from 2-A)

Loads both the pretrained GPT-2 and the SFT model, computes CP-LRP relevance for the same 20 SQuAD_v2 validation samples, saves delta = post − pre.

Visualizations:
- `compare_sample{0–4}.png`: 3-panel heatmap per sample (Pre-FT / Post-FT / Delta), wrapped-row layout
- `mean_delta.png`: bar chart of mean relevance change per token position across all 20 samples

**Output:** `aiaa4051/task2/comparison/`

---

### Task 3 — Parameter-Level Analysis (20 pts)

#### 3-A: Train Model A and Model B (GPU, ~4–8 h total)

```bash
nohup python task3_train_models.py > aiaa4051/logs/task3_train_models.log 2>&1 &
```

Trains two independent GPT-2 models from the same random initialization:
- **Model A** — fine-tuned on SciQ (`aiaa4051/data/sciq_train3000`, 3 epochs)
- **Model B** — fine-tuned on SQuAD_v2 (`aiaa4051/data/squad_v2_train3000`, 3 epochs)

**Output:**
- `aiaa4051/task3/modelA/final/`
- `aiaa4051/task3/modelB/final/`

---

#### 3-B: Parameter Relevance Comparison (local/GPU, ~20 min)

```bash
python task3_param_lrp.py
```

**Requires:** both model finals from 3-A.

For each of 3 test inputs, computes per-layer MLP relevance:
1. Registers forward hooks on each layer's `block.mlp.c_fc`
2. Runs CP-LRP forward+backward
3. Computes `R_layer = |x_mean @ W|.sum()` for each layer
4. Averages over 3 inputs and normalizes by total relevance

Generates:
- `param_relevance_comparison.png`: absolute + normalized side-by-side bar charts
- `param_relevance_diff.png`: difference bar chart (A − B), with interpretation of top-3 diverging layers

**Output:** `aiaa4051/task3/comparison/`

---

## 5. Results Summary

### Task 1

- Named entities and question keywords consistently receive the highest positive relevance (green).
- The `Answer:` delimiter token is strongly negative (red) — correctly captured as suppressive.
- Function words and punctuation are near-neutral.
- **Faithfulness AUC:** AttnLRP = 0.1359, Random = 0.1245. Non-monotonic curve shape due to signed relevance scores (see §6).

### Task 2

- Post-SFT model assigns substantially higher relevance to question-side and answer-boundary tokens, especially positions 31–33 and 58–59.
- Post-SFT model reduces relevance on mid-context background content.
- Named entities related to the answer (e.g., `Normandy`, `centuries`) gain stronger positive relevance after fine-tuning.

### Task 3

- Both models share the same macro-level layer profile: Layers 0, 2, 3, 10, 11 dominate; Layer 1 is weakest.
- **Key difference**: Model B (SQuAD_v2) relies ~1.7 percentage points more on Layer 11 (final layer) than Model A (SciQ).
- Model A (SciQ) distributes relevance slightly more towards mid-network Layers 3 and 10.
- Interpretation: span extraction (SQuAD_v2) demands more final-layer output-oriented processing; concept-level QA (SciQ) engages intermediate relational layers more.

---

## 6. Strengths & Limitations

### Strengths

| Aspect | Detail |
|---|---|
| Correct CP-LRP implementation | `stop_gradient` on Q and K; identity rule for MLP and LayerNorm; gradient flows through V path |
| Compatibility patch | Manual monkey-patching of `GPT2Attention._attn` cleanly handles `transformers 4.44.x` without modifying library source |
| Signed relevance | Preserves full sign information; positive = promotes, negative = suppresses |
| Full-scale execution | All 200 samples processed (not just a debug subset); confirmed by cluster logs |
| Meaningful findings | All three tasks produce interpretable, linguistically plausible results consistent with prior work |

### Limitations & Known Issues

#### L1 — Faithfulness metric mismatch with signed scores (Task 1)

The standard Pixel Flipping protocol sorts tokens by **descending** relevance. With signed CP-LRP scores, this masks all positive-relevance tokens first. Once the masking crosses into negative-relevance tokens, their removal *increases* model confidence (released inhibition), causing a non-monotonic confidence curve. The AttnLRP AUC (0.1359) thus appears *higher* (worse) than the random baseline (0.1245), which is misleading.

**Fix**: either (a) add a **worst-order baseline** (mask by *ascending* relevance) to bound the metric; or (b) switch to **Comprehensiveness** (drop in confidence when masking only top-K positive tokens) and **Sufficiency** (confidence using only top-K positive tokens), which are designed for signed attribution methods.

#### L2 — Parameter-level LRP is approximate (Task 3)

The current implementation computes `R_layer = |x_mean @ W|` where `x_mean` is the mean over the sequence dimension. This is not a strict LRP-ε propagation through the weight matrix (which requires token-level relevance scores already computed at each layer). It is a proxy that correlates weight magnitude with mean activation, which may over- or under-weight certain layers.

**Fix**: implement true LRP-ε by backpropagating relevance scores layer-by-layer, tracking `R[j] = sum_i (W[i,j] * x[j]) / (z + eps) * R[i]`.

#### L3 — Only 5 comparison visualizations saved (Task 2)

`task2_compare.py` computes results for 20 samples but only saves 5 PNG heatmaps (`for i in range(min(5, N))`). The PKL file contains all 20 samples.

**Fix**: change `min(5, N)` to `N` in the visualization loop.

#### L4 — No worst-order or sufficiency/comprehensiveness curves (Task 1)

Only two curves are plotted (relevance order + random). A complete faithfulness evaluation requires at least three curves to form meaningful upper/lower bounds.

#### L5 — Single test input per model in Task 3 (partially fixed)

The current code uses 3 test inputs and averages over them, which is an improvement over a single input but still a small sample. Results may be sensitive to input choice, especially for short science QA inputs that may not trigger all layers equally.

---

## 7. Improvement Roadmap

Listed in descending priority:

### P1 — Fix faithfulness evaluation (Task 1, ~2 h)

Add worst-order baseline and comprehensiveness/sufficiency scores:

```python
# In task1_faithfulness.py, add:
def worst_order_baseline(sample):
    """Mask by ascending relevance (least important first)."""
    sorted_idx = np.argsort(sample["relevance"]).tolist()  # ascending
    ...

def comprehensiveness(sample, k_frac=0.2):
    """Drop in confidence when masking top-K positive tokens only."""
    scores = sample["relevance"]
    pos_idx = np.where(scores > 0)[0]
    top_k = pos_idx[np.argsort(scores[pos_idx])[::-1][:int(len(pos_idx) * k_frac)]]
    # compare full confidence vs. confidence with top_k masked
    ...
```

### P2 — Generate all 20 comparison visualizations (Task 2, 5 min)

```python
# In task2_compare.py, line 108:
for i in range(N):   # was: range(min(5, N))
    plot_comparison(i)
```

### P3 — Implement true LRP-ε for parameter analysis (Task 3, ~1 day)

Replace the approximate `x_mean @ W` proxy with a proper layer-wise relevance propagation through MLP weights. Reference: Bach et al. (2015), Equation 11.

### P4 — Expand Task 3 test inputs

Use a representative set of 20+ inputs from both SQuAD_v2 and SciQ validation sets to get more stable per-layer relevance estimates. Average and report standard deviation to quantify stability.

### P5 — Add attention-head-level analysis (bonus)

Extend Task 1/2 to show per-head relevance distribution (which heads are most responsible for named entity attribution). This requires capturing per-head attention weights during the CP-LRP pass.

### P6 — Update model path handling

Replace hardcoded `/home/user/project/gpt2` paths with a configurable constant or CLI argument:

```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="gpt2")
args = parser.parse_args()
```

---

## 8. References

1. Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K.-R., & Samek, W. (2015). On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation. *PLOS ONE*, 10(7).

2. Achtibat, R., Hatefi, S. M. V., Dreyer, M., Jain, A., Wiegand, T., Lapuschkin, S., & Samek, W. (2024). AttnLRP: Attention-aware layer-wise relevance propagation for transformers. *ICML 2024*.

3. Samek, W., Binder, A., Montavon, G., Lapuschkin, S., & Müller, K.-R. (2017). Evaluating the visualization of what a deep neural network has learned. *IEEE Transactions on Neural Networks and Learning Systems*, 28(11), 2660–2673.

4. LRP-eXplains-Transformers (lxt) library: https://github.com/rachtibat/LRP-eXplains-Transformers
