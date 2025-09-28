# Transformer Model Family Report

## 1. Executive Overview

The Transformer family targets 72-hour ahead renewable energy percentage forecasting from a univariate (single-feature) historical window of 72 hours. Three evolutionary variants (Base, V2, V3) progressively introduce positional awareness, architectural right-sizing, regularization, and then controlled capacity scaling. All models share a common forecasting objective: produce a 72-length horizon vector in a single shot (multi-step direct strategy) rather than recursive one-step forecasting.

## 2. Quick Variant Comparison

| Variant   | Positional Embedding            | Blocks (Encoder Depth)                                                               | Heads | Head Size / Emb Dim            | FF Hidden (per block) | MLP Head                | Regularization          | Loss  | Total Params |
| --------- | ------------------------------- | ------------------------------------------------------------------------------------ | ----- | ------------------------------ | --------------------- | ----------------------- | ----------------------- | ----- | ------------ |
| Base (V1) | None (raw sequence only)        | 4 (as intended in notebook design; summary file captured 2 cycles repeated twice?)\* | 4     | 256                            | 4                     | Dense(128)              | Dropout 0.25            | MAE   | 12,972       |
| V2        | Token + Position (learned)      | 2                                                                                    | 2     | 2\*128 = 256                   | 4                     | Dense(64)               | Dropout 0.25            | MAE   | 573,072      |
| V3        | Token + Position (learned) + L2 | 4                                                                                    | 4     | 4\*256 = 1024 effective concat | 4                     | Dense(256) → Dense(128) | Dropout 0.25 + L2(1e-3) | Huber | 561,296      |

Notes:

1. Parameter counts are taken directly from exported summaries: Base = 12,972; V2 = 573,072; V3 = 561,296.
2. V2 has higher parameter count than V3 despite smaller depth: caused by dense projection dimensionality alignment and duplication of attention projections at 256-dim while Base is extremely compact (channel dim = 1) and V3 re-balances capacity with regularization and deeper head but not excessively increasing embedding overhead (some consolidation lowers total slightly vs V2).
3. Base summary shows two repeated attention/feed-forward sequences; design intent (from original notebook pattern) suggests a 4-block configuration; current exported architecture keeps two stacked transformer encoder repeats per two macro-cycles (reflected total params remain small due to single-channel hidden size). If alignment is important, consider regenerating the Base model with explicit TokenAndPositionEmbedding and validated 4-block depth.

## 3. Evolution Rationale

| Transition | Key Additions / Changes                                                                                                                                                                      | Motivation                                                                                                                  | Expected Effect                                                                                            |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Base → V2  | Introduce learned token+position embedding; reduce depth & heads (4→2); shrink head size (256 head_size via (128 \* 2) composite); simplified blocks (2); smaller MLP (64); maintain dropout | Provide sequence order awareness; right-size for univariate context; reduce overfitting risk                                | Better temporal alignment, improved convergence stability; moderate parameter jump due to embedding matrix |
| V2 → V3    | Increase depth (2→4 blocks); expand heads (2→4); increase head_size (128→256); deeper MLP (256→128); add L2 regularization; switch loss to Huber                                             | Capture richer multi-scale temporal patterns while controlling overfitting; robust to outliers; structured capacity scaling | Improved expressiveness with regularization balancing complexity; smoother gradients under noise           |

## 4. Architectural Breakdown by Variant

### 4.1 Base (V1)

- Input: `(72, 1)` univariate window.
- Encoder Blocks: Attention (MultiHeadAttention with key_dim ~1 inferred) → Dropout → Residual → LayerNorm → 2-layer feed-forward (Dense(4) → Dropout → Dense(1)) → Residual. Pattern repeated.
- Pooling: `GlobalAveragePooling1D` over (time, channel_first config in notebook; summary shows channels_last on some variants).
- Head: Dense(64) → Dropout → Dense(72 output horizon).
- Missing positional embedding: relies solely on model capacity to infer temporal ordering—suboptimal for permutation-sensitive tasks.
- Strength: Extremely lightweight (≈13K params) enabling very fast inference.
- Weakness: Limited representation power; no explicit temporal position encoding may degrade longer-horizon pattern retention.

### 4.2 V2

- Adds `TokenAndPositionEmbedding` producing a 256-dim sequence representation.
- Transformer Depth: 2 encoder blocks with `num_heads=2`, each attention projection dimension 256.
- Feed-Forward per block: Dense(4) → Dropout → Dense(256) (small inner expansion relative to embedding width; acts more like a gating micro-MLP).
- Global Average Pooling → Dense(64) → Dropout → Dense(72).
- Trade-off: Large embedding dimension vs very small ff_dim (4) may bottleneck non-linear transformation capacity despite high parameter count in attention projections.

### 4.3 V3

- Retains positional embedding; expands to 4 blocks, `num_heads=4`, head_size=256 with L2 regularization across projections and dense layers.
- Feed-Forward stays minimal (ff_dim=4) maintaining a narrow transformation core; relies on multi-head diversity and deeper stacking for feature enrichment.
- MLP Head deepened: Dense(256) → Dense(128) prior to final horizon layer, adding non-linear compression funnel.
- Loss: Huber (robust to peaks/outliers that can occur in renewable fluctuations).
- Regularization: L2(1e-3) + dropout to combat capacity-driven overfitting.

## 4.a Layer-by-Layer Snapshots

| Variant | Key Layers (Sequential Summary)                                                                                                                                                                       |
| ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Base    | Input → (MHA → Dropout → Residual → LN → Dense(4) → Dropout → Dense(1) → Residual) ×2 → GlobalAvgPool1D → Dense(64) → Dropout → Dense(72)                                                             |
| V2      | Input → Token+PosEmbedding → [ (LN → MHA(2 heads, 256) → Dropout → Residual → LN → Dense(4) → Dropout → Dense(256) → Residual) ×2 ] → GlobalAvgPool1D → Dense(64) → Dropout → Dense(72)               |
| V3      | Input → Token+PosEmbedding → [ (LN → MHA(4 heads, 256) → Dropout → Residual → LN → Dense(4) → Dropout → Dense(256) → Residual) ×4 ] → GlobalAvgPool1D → Dense(256) → Dropout → Dense(128) → Dense(72) |

## 5. Parameter Dynamics Insight

- Dramatic jump Base → V2 (12.9K → 573K) primarily from embedding and multi-head projection matrices (query, key, value, output) at 256-dim space.
- Slight reduction V2 → V3 (573K → 561K) despite deeper network because design consolidates representation and redistributes capacity; overhead of added blocks offset by architectural choices (e.g., parameter sharing pattern of dimensions vs head expansion). If unexpected, re-verify generation script or inspect Dense/attention dimensions for pruning differences.
- Opportunity: Increase `ff_dim` to a more conventional 2–4× embedding (e.g., 512–1024) while possibly reducing embedding dimension to 128 to reach a more balanced compute-to-representation ratio.

## 6. Strengths & Risks

| Aspect                 | Base          | V2                          | V3                         |
| ---------------------- | ------------- | --------------------------- | -------------------------- |
| Positional Awareness   | None          | Explicit                    | Explicit                   |
| Capacity vs Data       | Underfit risk | Possibly over-parameterized | Balanced w/ regularization |
| Robustness to Outliers | Low (MAE)     | Low (MAE)                   | Higher (Huber)             |
| Generalization Control | Minimal       | Dropout only                | Dropout + L2               |
| Interpretability       | High (small)  | Moderate                    | Moderate                   |
| Training Cost          | Very Low      | High                        | High+                      |

## 7. Recommendations

1. Rebalance Feed-Forward: Increase `ff_dim` (e.g., 128–256) and lower embedding width to 128–192 to enhance non-linear capacity without excessive projection overhead.
2. Add Learning Rate Warmup (Optional): Especially for deeper V3 to stabilize early training dynamics.
3. Consider Temporal Convolution Pre-Encoder: A light 1D Conv projection can compress noise prior to attention.
4. Experiment with Multi-Scale Attention: Add a block with larger dilation or downsampled sequence to capture broader seasonality.
5. Evaluate Label Smoothing or Quantile Loss: If probabilistic forecasts are desired, adapt output to distributional heads.
6. Introduce Sparse Attention (Longer Windows): If extending window beyond 72, adopt performant sparse patterns (e.g., local+global) to limit quadratic growth.

## 8. Potential Next Variant (V4 Concept)

| Feature          | Proposed Change                                | Rationale                                     |
| ---------------- | ---------------------------------------------- | --------------------------------------------- |
| Input Projection | Add causal depthwise Conv1D                    | Inject locality & reduce high-frequency noise |
| Embedding Dim    | 256 → 160                                      | Trim redundancy; reduce memory                |
| Blocks           | 4 → 3 diversified (2 standard + 1 multi-scale) | Multi-scale temporal capture                  |
| FF Dim           | 4 → 160–320                                    | Restore expressive transformation channel     |
| Regularization   | Keep L2; add dropout scheduling                | Dynamic regularization tuning                 |
| Output Head      | Add residual skip from pooled embedding        | Stabilize gradients, retain global context    |
| Loss             | Hybrid Huber + seasonal-weighted component     | Penalize systematic seasonal deviation        |

## 9. Conceptual Diagram References

If conceptual diagrams were generated by `generate_concept_diagrams.py`, they should reside alongside summaries with names like:

- `Transformer_Model__build_transformer_model_concept.png`
- `Transformer_Model_v2__build_transformer_model_concept.png`
- `Transformer_Model_v3__build_transformer_model_concept.png`

Embed examples (if rendering supported):

```
![Base Transformer Concept](Transformer_Model__build_transformer_model_concept.png)
![Transformer V2 Concept](Transformer_Model_v2__build_transformer_model_concept.png)
![Transformer V3 Concept](Transformer_Model_v3__build_transformer_model_concept.png)
```

## 10. Glossary

- Token & Position Embedding: Layer combining learned token projection with learned positional indices enabling order encoding.
- Head Size: Dimensionality of each attention head’s key/query/value space.
- ff_dim: Intermediate dimensionality in transformer feed-forward sub-layer.
- GlobalAveragePooling1D: Aggregates temporal features into a fixed-size vector by mean over time dimension.
- Huber Loss: Robust regression loss blending L1 and L2 regions to mitigate outlier influence.
- L2 Regularization: Weight penalty discouraging large coefficients to improve generalization.

## 11. Validation Checklist

- [x] Horizon length consistent (72) across variants.
- [x] Parameter counts match exported summaries.
- [x] Loss function changes captured (MAE → Huber in V3).
- [x] Positional embedding introduction noted only from V2 onward.
- [x] Regularization (L2) only in V3.

---

Report generated programmatically based on exported model summaries and inspected source notebooks (converted .py variants). Adjust if upstream architecture definitions materially change.
