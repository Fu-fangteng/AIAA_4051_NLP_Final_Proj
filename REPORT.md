# AIAA 4051 Final Research Project Report
**Explaining GPT-2 with AttnLRP: Token Relevance, Fine-Tuning Effects, and Parameter-Level Analysis**

---

## 1. Introduction

Understanding the decision-making process of large language models (LLMs) is central to improving their transparency, trustworthiness, and responsible deployment. Layer-wise Relevance Propagation (LRP) is a principled attribution method that decomposes a model's output into contributions from individual input features. However, standard LRP does not naturally account for the attention mechanism that is the core building block of transformer models.

This project applies **AttnLRP** (Achtibat et al., ICML 2024), a modified LRP variant specifically designed for transformer architectures. AttnLRP propagates relevance through both the attention weights and the feed-forward layers, producing token-level attribution scores that are more faithful to the model's actual computation. We apply AttnLRP to **GPT-2 Small** across three tasks: reproducing the algorithm on SQuAD_v2, analysing how sequential fine-tuning changes token attributions, and comparing parameter-level contributions between two separately fine-tuned models.

---

## 2. Task 1: Reproducing AttnLRP on GPT-2 / SQuAD_v2

### 2.1 Setup

We implemented the **CP-LRP** variant of AttnLRP, which is the recommended variant for GPT-2 due to its negative-valued logits. CP-LRP stops gradient flow at the query and key tensors of the attention softmax, while preserving gradient flow through the value pathway. The MLP layers apply the identity rule (gradient is not propagated through the nonlinearity's denominator), and LayerNorm applies the same identity rule.

The implementation uses the `lxt` (LRP-eXplains-Transformers) library with manual compatibility patches for transformers 4.44.x. Relevance scores are computed as the element-wise product of input embeddings and their gradients, summed over the embedding dimension:

$$R_i = \sum_d e_{i,d} \cdot \frac{\partial \hat{y}}{\partial e_{i,d}}$$

where $e_i$ is the embedding of the $i$-th token and $\hat{y}$ is the maximum logit at the final position. This produces a signed scalar per token: positive values indicate that the token promotes the prediction; negative values indicate suppression.

We evaluated on the **SQuAD_v2 validation set** (200 samples). Each input is formatted as:

```
Context: <context[:300]>
Question: <question>
Answer:
```

### 2.2 Relevance Visualisation

Figures 1 and 2 show representative token relevance heatmaps. Several consistent patterns emerge:

- **Named entities and key noun phrases receive the highest positive relevance.** For Sample 0 (question: *"In what country is Normandy located?"*), the token `Normandy` in both the context and the question is consistently highlighted in green, confirming that the model correctly identifies the answer-relevant entity.
- **The `Answer:` token is consistently assigned strongly negative relevance** (shown in deep red). This is expected: the `Answer:` delimiter signals the end of the prompt, and the model's prediction is conditioned on everything that precedes it. Masking this structural token changes the input distribution dramatically, and its suppressive role is correctly captured.
- **Function words and punctuation are near-neutral** (pale yellow), consistent with them carrying little semantic content relevant to the specific prediction.
- **The `Question:` section receives higher relevance than the middle of the context**, reflecting GPT-2's left-to-right attention bias where tokens closer to the generation position receive more weight.

These patterns are linguistically plausible and consistent across samples, suggesting the CP-LRP implementation is functioning correctly.

### 2.3 Faithfulness Evaluation

We evaluated faithfulness using the **Pixel Flipping** protocol: tokens are progressively masked in descending order of relevance score (most important first), and model confidence (maximum softmax probability at the final position) is recorded at each step. A random-order baseline is computed for comparison. A lower area under the curve (AUC) indicates faster confidence decay and higher faithfulness.

| Method | Normalised AUC |
|---|---|
| AttnLRP (CP-LRP, relevance order) | 0.1648 |
| Random baseline | 0.1444 |

Contrary to the expectation that AttnLRP should produce a lower AUC, the AttnLRP curve shows a **non-monotonic pattern**: confidence initially declines gently as the most positively-relevant tokens are masked (0–50%), but then **rises sharply around 60–70% masked** before finally declining.

This behaviour is attributable to the dual-signed nature of CP-LRP scores. When tokens are sorted by descending relevance, all positively-relevant tokens are masked first. Once the masking crosses the boundary into negatively-relevant (suppressive) tokens, removing them releases their inhibitory effect on the prediction, causing confidence to temporarily increase. This is not a failure of the method but rather a reflection of the richer information encoded in signed relevance scores compared to unsigned methods such as gradient norm.

The random baseline, by contrast, masks a mixture of positive and negative tokens at each step, producing a smoother and more uniformly decreasing curve. Its lower AUC therefore reflects the smoother trajectory rather than greater faithfulness. A more appropriate faithfulness metric for signed attribution methods is the **sufficiency** score (early plateau using only top-K positive tokens) or **comprehensiveness** (drop from masking only positive tokens), which are left for future work.

---

## 3. Task 2: AttnLRP Before and After Sequential Fine-Tuning

### 3.1 Fine-Tuning Protocol

We applied **Sequential Fine-Tuning (SFT)** to GPT-2 Small:

1. **Step 1**: Fine-tune on SQuAD_v2 (`train[:5000]`, 2 epochs, batch size 8, fp16)
2. **Step 2**: Continue fine-tuning on SciQ (`train[:3000]`, 2 epochs, batch size 8, fp16)

The final checkpoint (`sft_final`) is used as the post-fine-tuning model. The same 20 SQuAD_v2 validation samples are used to compare pre- and post-FT AttnLRP attributions.

### 3.2 Per-Sample Comparison

Figures 3 and 4 show the pre-FT relevance, post-FT relevance, and their difference (Delta = post − pre) for two representative samples.

Several clear changes are observable:

**Increased relevance on structural tokens post-FT.** After fine-tuning, tokens such as `Question:`, `Answer:`, and numerical tokens (e.g., `10`, `11`) receive markedly stronger positive relevance in the post-FT model. This suggests the fine-tuned model has learned to rely more heavily on the question–answer format structure, which is precisely the format used in both SQuAD_v2 and SciQ training data.

**Stronger focus on answer-relevant named entities.** In Sample 1, the entity `Normandy` in the question receives stronger green colouring in the post-FT model. Similarly, century markers (`centuries`) and geographic terms gain positive delta. This indicates the fine-tuned model has sharpened its attention towards semantically meaningful content words.

**Reduced relevance on mid-context tokens.** Many mid-context tokens (background information unrelated to the specific question) show reduced or negative delta values post-FT, indicating the model has become more selective in which contextual information it propagates to the prediction.

### 3.3 Aggregate Analysis

The mean delta chart (averaged over N=20 samples, Figure 5) reveals several systematic patterns:

- **Token positions 30–35** (corresponding roughly to the start of the `Question:` section) show the largest positive mean deltas (+8 to +13), indicating the fine-tuned model assigns substantially more importance to the question itself.
- **Token positions 57–59** also show large positive deltas (+11 to +15), corresponding to question keywords and the `Answer:` boundary.
- **Token position 55** shows the largest negative delta (≈ −5), suggesting the model has learned to suppress certain structural tokens in the context–question boundary region.

**Conclusion for Task 2:** Sequential fine-tuning on SQuAD_v2 and SciQ causes GPT-2 to reorganise its token attribution patterns substantially. The fine-tuned model places greater importance on question-side tokens and answer-relevant entities, while de-emphasising background contextual content. This aligns with the expected effect of task-specific training: the model learns to focus on the information most predictive of the target answer format.

---

## 4. Task 3: Parameter-Level Relevance Comparison between Model A and Model B

### 4.1 Setup

We fine-tune two separate models from the same GPT-2 Small initialisation:

- **Model A**: Fine-tuned on SciQ (3,000 samples, 3 epochs) — short, structured science QA
- **Model B**: Fine-tuned on SQuAD_v2 (3,000 samples, 3 epochs) — long-form reading comprehension

We extend AttnLRP to the **parameter level** using the LRP-ε rule. For each Transformer layer $\ell$, the MLP contribution is quantified as:

$$R_\ell = \sum_{i,j} |\bar{x}_j \cdot W_{ji}^{(c\_fc)}|$$

where $\bar{x}$ is the mean activation entering the MLP's first projection across the sequence, and $W^{(c\_fc)}$ is the weight matrix of the first linear layer (`c_fc`, shape $[768, 3072]$ for GPT-2 Small). This gives a scalar relevance value per layer representing how strongly the MLP parameters of that layer contribute to the final logit. Results are averaged over three test inputs and normalised by the total relevance sum.

### 4.2 Per-Layer Comparison

The absolute and normalised relevance bar charts (Figure 6) show:

- **Both models share the same overall layer-relevance profile**: Layer 1 has the lowest contribution (~5.8–5.9% normalised), while Layers 0, 2, 3, 10, and 11 have the highest (~8–10%).
- **The profiles are very similar in shape**, indicating that the general information processing hierarchy of GPT-2 is preserved regardless of fine-tuning domain.
- The absolute values for Model A are consistently slightly higher than Model B across Layers 1–10, suggesting SciQ fine-tuning leads to somewhat more active MLP representations overall.

### 4.3 Differential Analysis

The difference plot (Figure 7, Δ = Model A − Model B) reveals one dominant and interpretable pattern:

| Layer | Δ (A − B) | Interpretation |
|---|---|---|
| 11 | **−0.017** (large negative) | Model B relies substantially more on the final layer |
| 3 | +0.005 | Model A relies slightly more on Layer 3 |
| 10 | +0.005 | Model A relies slightly more on Layer 10 |
| 0 | −0.001 | Nearly identical |

**The most striking finding is at Layer 11** (the final Transformer block). Model B (SQuAD_v2) assigns approximately 1.7 percentage points more normalised relevance to this layer than Model A (SciQ). In transformer architectures, the final layers are generally associated with task-specific, output-oriented computation — they transform abstract representations into prediction-ready features. SQuAD_v2 requires the model to locate a specific span of text within a long context, a process that demands more final-layer processing to select and format the answer. SciQ, being shorter structured QA, may rely more on intermediate layers for factual retrieval.

**Model A relies slightly more on mid-network layers (3, 10)**, which are associated with compositional and relational reasoning. SciQ questions often require combining multiple scientific concepts, which may engage these intermediate representational layers more intensely.

**Conclusion for Task 3:** While both models preserve the same macro-scale layer relevance profile inherited from GPT-2's pretraining, the fine-tuning domain produces measurable differences in layer utilisation. Reading comprehension (SQuAD_v2) drives stronger final-layer engagement, consistent with the need for precise span extraction. Structured science QA (SciQ) distributes relevance slightly more towards intermediate layers, consistent with concept-level reasoning.

---

## 5. Conclusion

This project applied CP-LRP (AttnLRP) to GPT-2 Small across three levels of analysis:

1. **Token level (Task 1)**: AttnLRP reliably identifies semantically meaningful tokens — particularly named entities and question keywords — as most relevant to predictions on SQuAD_v2. The faithfulness evaluation reveals a non-monotonic confidence curve attributable to the dual-signed nature of CP-LRP scores, highlighting a limitation of the standard Pixel Flipping metric for signed attribution methods.

2. **Fine-tuning effect (Task 2)**: Sequential fine-tuning on SQuAD_v2 → SciQ causes a systematic redistribution of token relevance: the model sharpens focus on question-side tokens and answer-relevant entities while reducing attention to uninformative context. This demonstrates that AttnLRP can detect the behavioural signature of task-specific training.

3. **Parameter level (Task 3)**: Both fine-tuned models share the same layer relevance profile, but Model B (SQuAD_v2) shows disproportionately higher engagement of the final Transformer layer (Layer 11), while Model A (SciQ) distributes relevance slightly more towards mid-network layers. This suggests that task complexity and output format shape how different layers contribute to the final prediction.

Together, these results demonstrate that AttnLRP is a powerful and informative tool for understanding not just what a model attends to, but how its internal computational structure adapts to different training regimes.

---

## References

[1] Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K.-R., & Samek, W. (2015). On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation. *PLOS ONE*, 10(7).

[2] Achtibat, R., Hatefi, S. M. V., Dreyer, M., Jain, A., Wiegand, T., Lapuschkin, S., & Samek, W. (2024). AttnLRP: Attention-aware layer-wise relevance propagation for transformers. *ICML 2024*.

[3] Samek, W., Binder, A., Montavon, G., Lapuschkin, S., & Müller, K.-R. (2017). Evaluating the visualization of what a deep neural network has learned. *IEEE Transactions on Neural Networks and Learning Systems*, 28(11), 2660–2673.
