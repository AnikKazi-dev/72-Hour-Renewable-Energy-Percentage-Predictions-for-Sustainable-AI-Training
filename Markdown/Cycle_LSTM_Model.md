# Cycle LSTM Model Family Report

## 1. Overview

The Cycle LSTM family targets 72‑hour ahead renewable percentage forecasting using recurrent sequence modeling over a 72‑step (3 day) look‑back. All variants predict a 72‑hour horizon. Progression (Base → V2 → V3) follows a typical refinement arc:

- Base: Custom bidirectional (cycle) style via forward + backward LSTM sum.
- V2: Cleaner standardized Bidirectional wrappers + stacking for hierarchical temporal abstraction.
- V3: Capacity + robustness upgrade (larger hidden sizes, regularization, normalization, robust loss).

Design goals:

- Capture medium‑range temporal dependencies without attention overhead.
- Balance accuracy vs. parameter budget.
- Provide upgrade path with controlled complexity increases.

## 2. Variant Summary

| Variant | Params | Core Recurrent Structure                                                         | Dense Head                                   | Loss  | Regularization & Stabilizers        | Key Differences                                                                                      |
| ------- | ------ | -------------------------------------------------------------------------------- | -------------------------------------------- | ----- | ----------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Base    | 158.9K | LSTM(128 fwd) + LSTM(128 bwd emulated via `go_backwards=True`) aggregated by Add | Dense(128) → Dropout → Dense(72)             | MAE   | Dropout(0.3)                        | Manual cycle (add) instead of standard bidirectional; single depth recurrent stage                   |
| V2      | 152.8K | Stacked Bidirectional(LSTM 64) → Bidirectional(LSTM 64)                          | Dense(100) → Dropout → Dense(72)             | MAE   | Dropout(0.3)                        | Uses official Bidirectional wrapper; stacked temporal hierarchy; slightly fewer params despite stack |
| V3      | 570.1K | Stacked Bidirectional(LSTM 128) → Bidirectional(LSTM 128)                        | Dense(128) → BatchNorm → Dropout → Dense(72) | Huber | Dropout(0.3), L2 (0.001), BatchNorm | Large capacity jump; robust loss + L2; normalization added                                           |

Notes:

- V2 parameter count is slightly lower than Base due to narrower (64‑unit) stacked layers vs. Base’s wider 128/128 pair combined by addition.
- V3 more than triples parameters to expand representational capacity for complex seasonal–short term interplay.

## 3. Architectural Breakdown

### 3.1 Base Variant

Sequence Flow:

1. Input (72×1) → LSTM(128, forward)
2. Parallel path: LSTM(128, backward) over same input (`go_backwards=True`)
3. Add() combines forward & backward latent (128 units)
4. Dense(128, ReLU) → Dropout(0.3)
5. Dense(72) → 72‑hour forecast vector

Characteristics:

- Simple two‑direction fusion by additive merge (rather than concat) keeps dimensionality fixed at 128.
- No explicit normalization; relies on scaling + dropout only.
- Suitable as a light baseline recurrent forecaster.

### 3.2 V2 Variant (Refined)

Sequence Flow:

1. Input → Bidirectional(LSTM 64, return_sequences=True) → outputs (72×128)
2. Bidirectional(LSTM 64, return_sequences=False) → 128 latent summary
3. Dense(100, ReLU) → Dropout(0.3)
4. Dense(72)

Enhancements vs Base:

- Official Bidirectional wrapper improves clarity & consistency.
- Stacked architecture (return_sequences=True in first layer) enables hierarchical temporal feature extraction (shorter motifs → aggregated representation).
- Slight reduction in params while increasing abstraction depth.

### 3.3 V3 Variant (Scaled & Regularized)

Sequence Flow:

1. Input → Bidirectional(LSTM 128, return_sequences=True, L2=0.001)
2. Bidirectional(LSTM 128, return_sequences=False, L2=0.001) → 256 latent → Dense(128, ReLU, L2=0.001)
3. BatchNormalization → Dropout(0.3)
4. Dense(72)

Enhancements vs V2:

- Capacity scaling (64→128 units; concatenated bidirectional width 256 before head).
- L2 regularization on recurrent & dense weights counters overfitting risk from increased width.
- BatchNormalization stabilizes activations post-dense layer.
- Huber loss provides robustness to outliers/spikes in grid variability.

## 4. Layer-by-Layer Snapshots

| Variant | Sequential Summary                                                                                         |
| ------- | ---------------------------------------------------------------------------------------------------------- |
| Base    | Input → LSTM(128,fwd) + LSTM(128,bwd) → Add → Dense(128) → Dropout → Dense(72)                             |
| V2      | Input → BiLSTM(64, seq) → BiLSTM(64, summary) → Dense(100) → Dropout → Dense(72)                           |
| V3      | Input → BiLSTM(128, seq, L2) → BiLSTM(128, summary, L2) → Dense(128, L2) → BatchNorm → Dropout → Dense(72) |

## 5. Comparative Analysis

### 5.1 Capacity vs Generalization

- Base: Lowest depth; may underfit complex long seasonal transitions.
- V2: Better temporal abstraction via stacking; still efficient.
- V3: Highest expressiveness; monitor validation gap (watch for divergence if dataset size limited).

### 5.2 Parameter Efficiency

- V2 achieves comparable (often superior) representational depth to Base with fewer parameters through narrower stacked layers.
- V3 introduces a large jump; deploy only if empirical gains (MAE / RMSE / R²) justify carbon & latency overhead.

### 5.3 Robustness Features

| Feature                  | Base              | V2             | V3             | Benefit                  |
| ------------------------ | ----------------- | -------------- | -------------- | ------------------------ |
| Bidirectional Processing | Manual (add)      | Native wrapper | Native wrapper | Clear temporal context   |
| Stacking Depth           | 1 effective layer | 2              | 2 (wider)      | Hierarchical abstraction |
| Regularization (Dropout) | Yes               | Yes            | Yes            | Mitigates overfit        |
| L2 Weight Decay          | No                | No             | Yes            | Penalizes large weights  |
| Normalization            | No                | No             | BatchNorm      | Stabilizes training      |
| Robust Loss              | MAE               | MAE            | Huber          | Outlier resilience       |

### 5.4 When to Choose

| Scenario                    | Recommended Variant | Rationale                                              |
| --------------------------- | ------------------- | ------------------------------------------------------ |
| Fast baseline / low compute | Base                | Simplicity; acceptable for quick benchmarking          |
| Balanced accuracy vs cost   | V2                  | Deeper temporal modeling without large param inflation |
| Maximum accuracy attempt    | V3                  | Highest capacity + regularization stack                |
| Data noisy / spiky          | V3                  | Huber + L2 + BN resilience                             |
| Limited training samples    | V2                  | Avoids overfitting risk of V3                          |

### 5.5 Upgrade Guidance

- Try V2 first; escalate to V3 only if validation metrics plateau.
- If V3 overfits: raise L2 (0.001→0.002), add dropout in first BiLSTM output, or early stop patience reduction.
- Consider switching MAE→Huber in Base/V2 if outliers frequent.

## 6. Recommendations

| Goal                           | Action                                                                                                    |
| ------------------------------ | --------------------------------------------------------------------------------------------------------- |
| Reduce training carbon         | Use V2 with early stopping & lower batch size exploration.                                                |
| Improve long-horizon stability | Add learning rate scheduler (ReduceLROnPlateau) to Base (already in V2/V3 patterns).                      |
| Enhance interpretability       | Log intermediate BiLSTM embeddings (project to 2D via PCA for drift inspection).                          |
| Probabilistic forecasting      | Swap final loss to quantile (pinball) in V2/V3 for P50/P90 intervals.                                     |
| Deployment efficiency          | Distill V3 to a smaller student (64‑unit BiLSTM stack) via knowledge distillation on generated forecasts. |

## 7. Appendix

Parameter counts sourced from exported model summaries in `Model_Architectures/`.
All variants share: LOOK_BACK=72, HORIZON=72, feature count=1, seasonal segmentation (Winter/Summer) handled upstream.

---

Generated automatically. Adjust wording if integrating into paper draft.
