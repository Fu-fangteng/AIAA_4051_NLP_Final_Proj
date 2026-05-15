# AttnLRP on GPT-2 Small
**AIAA 4051 — NLP Final Project**  
Yenchi Tseng · Fangteng Fu · HKUST(GZ)

Applying CP-LRP (AttnLRP) to GPT-2 Small across three levels of analysis: token attribution, fine-tuning effects, and parameter-level layer contributions.

---

## Tasks

| | Description |
|---|---|
| **Task 1** | Token-level CP-LRP attribution on SQuAD\_v2 (first 200 validation examples); faithfulness evaluated via Pixel Flipping |
| **Task 2** | Compare CP-LRP attributions before and after sequential fine-tuning (SQuAD\_v2 → SciQ) on 20 samples |
| **Task 3** | Train two independent models (SciQ / SQuAD\_v2) and compare per-layer MLP parameter relevance over 1,000 evaluation inputs each |

---

## Setup

```bash
pip install torch transformers==4.44.2 datasets matplotlib seaborn numpy tqdm
cd LRP-eXplains-Transformers && pip install -e . && cd ..
```

> `transformers 4.44.x` is required — the CP-LRP attention patch targets `GPT2Attention._attn` in this version.

---

## Running

All scripts must be run from the project root. Scripts automatically create output directories under `aiaa4051/`.

```bash
# Task 1
python src/task1/task1_attnlrp.py          # compute relevance scores
python src/task1/task1_faithfulness.py     # Pixel Flipping evaluation
python src/task1/task1_visualization.py    # render heatmaps

# Task 2
python src/task2/task2_sft.py              # sequential fine-tuning (GPU)
python src/task2/task2_compare.py          # pre/post attribution comparison
python src/task2/task2_visualization.py    # render figures

# Task 3
python src/task3/task3_train_models.py     # train Model A (SciQ) + Model B (SQuAD_v2)
python src/task3/task3_param_lrp.py        # parameter relevance computation
python src/task3/task3_visualization.py    # render figures
```

---

## Results

**Task 1 — Faithfulness (Pixel Flipping, 200 samples)**

| Deletion order | Normalised AUC |
|---|---|
| AttnLRP (most relevant first) | 0.0190 |
| Random baseline | 0.0490 |
| Least-relevant-first | 0.0871 |

AttnLRP achieves the lowest AUC, confirming faithfulness. The curve is non-monotonic due to CP-LRP's dual-signed scores.

**Task 2 — Relevance shift after fine-tuning (20 samples)**

| Segment | Mean ΔR |
|---|---|
| Question | +1.67 |
| Context | +0.27 |

Fine-tuning systematically shifts relevance toward question-side tokens (~6× more than context).

**Task 3 — Parameter-level relevance (1,000 inputs each)**

| Layer | Δ (SciQ − SQuAD\_v2) | Higher model |
|---|---|---|
| 10 | +0.0196 | SciQ |
| 9 | +0.0130 | SciQ |
| 6 | −0.0126 | SQuAD\_v2 |

SciQ engages upper MLP layers more (parametric memory); SQuAD\_v2 concentrates relevance in middle layers (context processing).

---

## Report

See `report.tex` / `NLP_report.pdf` for full methodology and discussion.
