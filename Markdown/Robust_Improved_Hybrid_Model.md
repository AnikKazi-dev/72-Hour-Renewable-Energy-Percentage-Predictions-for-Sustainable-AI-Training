# Robust Improved Hybrid Model Family Report

## 1. Overview
The Robust Improved Hybrid family fuses multiple inductive biases for 72‑hour renewable energy forecasting:
- Local pattern extraction (temporal convolutions / depthwise separable convs in V2)
- Sequence memory (BiLSTM, GRU fusion in V2)
- Global dependency modeling (stacked Multi-Head Attention blocks)
- Linear decomposition (DLinear-style seasonal + trend heads; detrending enhanced in V2)
- Adaptive gating (learned sigmoid gate; temperature scaling + squeeze‑excite in V2)

Goal: Balance interpretability (linear decomposition) with expressiveness (deep contextualized latent) while controlling overfitting through architectural regularization.

## 2. Variant Summary
| Variant | Params | Key Additions / Differences | Fusion & Decomposition | Regularization Features |
|---------|--------|-----------------------------|------------------------|--------------------------|
| Base    | 203.68K | CNN→BiLSTM + Attention + Linear dual head + gate | Gate(deep vs. linear) | Dropout (0.1–0.2), LayerNorm |
| V2      | 306.74K | Positional encoding, depthwise residual CNN blocks, BiLSTM + GRU additive fusion, stacked smaller attention (pre-norm), squeeze‑excite, moving‑average detrending, temperature-scaled gating, optional quantile loss | Gate(deep vs. linear after detrend) | Dropout, LayerNorm, Squeeze‑Excite, MA detrend, Temperature gating |

## 3. Architectural Breakdown
### 3.1 Base Variant
Framework:
- Input (72×1) → Conv1D(32) + LN → (MHA + residual) × 2 with FFN sublayers
- Parallel Deep Path: Additional Conv1D(64) ×2 → BiLSTM(64) → BiLSTM(32) → GlobalAveragePooling (attention path)
- Concatenate(deep temporal rep, pooled attention rep)
- Dense(128) → Dropout → Dense(72) = Deep Forecast
- Linear Path: Flatten(Input) → Dense(72 seasonal) + Dense(72 trend) → Add = Linear Forecast
- Gated Fusion: gate = sigmoid(Dense(latent)); Output = gate * Deep + (1 − gate) * Linear

Notable Characteristics:
- Dual representation (contextual vs. linear) improves robustness under distribution shifts.
- Shallow attention stack limits parameter growth while granting global context.

### 3.2 V2 Variant (Enhanced)
Additions / Modifications vs Base:
- Positional Encoding: Concatenated sine/cosine improves temporal position awareness.
- Depthwise Residual Blocks: Two depthwise + pointwise Conv sequences with residual adds reduce parameters per receptive field gain.
- Recurrent Fusion: BiLSTM(64) + GRU(128) outputs added (broadens temporal dynamic capture) instead of stacked BiLSTMs.
- Attention Blocks: Two pre-norm stacks (LayerNorm → MHA → residual → LN → FeedForward (smaller key_dim) → residual) over widened channel dimension (128).
- Squeeze‑Excite (SE): Channel recalibration after attention stack.
- Detrended Linear Branch: MovingAverage(window=7) to produce trend; residual (input − MA) flattened → seasonal head + trend head with dropout on seasonal path.
- Non-linear Deep Branch Head: Dense(128) → Dropout → Dense(72) retained, but gating temperature scaling (divide logits by temperature before sigmoid) sharpens control.
- Optional Quantile Loss: Switch to pinball (median by default) for probabilistic forecasting.

Effect Summary:
- Capacity Increase: +~103K params primarily from wider attention (128 channels) and recurrent fusion.
- Robustness: MA detrending + SE + depthwise convs reduce overfitting risk for noisy seasonal patterns.
- Flexibility: Quantile loss support broadens use to probabilistic settings.

## 3.a Layer-by-Layer Snapshots
| Variant | Key Layers (Sequential Summary) |
|---------|---------------------------------|
| Base | Input → Conv1D(32) → LN → (MHA→Residual) ×2 → Conv1D(64) → Conv1D(64) → BiLSTM(64) → BiLSTM(32) → GlobalAvgPool (attention path) || Linear Path: Flatten → Dense(72 seasonal) + Dense(72 trend) → Add || Gating: Dense→Sigmoid blend |
| V2 | Input → PosEncoding → (DepthwiseConv→PointwiseConv→Residual) ×2 → (PreNorm: LN→MHA(32 key_dim, width 128)→Residual→LN→FFN(small)→Residual) ×2 → Squeeze-Excite → Split: Recurrent (BiLSTM(64)+GRU(128) add) & Attention pooled → Concat → Dense(128) → Dropout → Dense(72) || Linear Path: MA(window=7) trend + Residual seasonal → (Dense(72 trend) + Dense(72 seasonal with Dropout)) → Add || Gate: Dense / temperature-scaled Sigmoid |

## 4. Layer Spotlight
| Feature | Base Realization | V2 Realization | Benefit |
|---------|------------------|----------------|---------|
| Local Conv | Standard Conv1D (causal) | Depthwise + Pointwise residual blocks | Efficiency + richer local features |
| Recurrent Core | BiLSTM stack (64→32) | BiLSTM(64) + GRU(128) additive | Diverse temporal dynamics |
| Attention | 2× MHA (32-dim) | 2× MHA pre-norm (key_dim 32, width 128) | Stronger context with normalization stability |
| Decomposition | Flatten → dual Dense | MA detrend → residual seasonal + trend | Cleaner seasonal/trend separation |
| Gating | Sigmoid(Dense) | Temperature-scaled sigmoid | Tunable blend sharpness |
| Channel Recalibration | — | Squeeze‑Excite | Adaptive emphasis of informative channels |
| Positional Encoding | — | Sin/Cos concat | Temporal position awareness |
| Probabilistic Output | — | Optional quantile loss | Distributional forecasting |

## 5. Comparative Analysis
### 5.1 Capacity vs. Overfitting
- Base: Balanced; good starting point for moderate complexity datasets.
- V2: Higher ceiling; must monitor validation MAE / quantile calibration (if enabled).

### 5.2 Interpretability
- Both retain explicit linear seasonal + trend heads (easier to audit vs. monolithic deep nets).
- V2’s detrending clarifies linear component attribution.

### 5.3 Efficiency Considerations
| Variant | Relative Inference Cost (Approx) | Contributors |
|---------|----------------------------------|--------------|
| Base    | 1× (reference)                   | Moderate conv + BiLSTM + 2 MHA |
| V2      | ~1.4–1.6×                        | Added GRU path, wider attention, SE overhead |

### 5.4 When to Choose
| Scenario | Recommended Variant | Rationale |
|----------|---------------------|-----------|
| Fast baseline / limited compute | Base | Lower latency, simpler recurrent core |
| Noisy seasonal data | V2 | Detrending + SE + depthwise robustness |
| Need probabilistic forecasts | V2 | Quantile loss support |
| Capacity-limited edge device | Base | Fewer parameters & ops |
| Accuracy priority in varied regimes | V2 | Broader representational mix |

## 6. Potential Future Enhancements
- Dynamic Kernel Selection: Learnable or adaptive MA window.
- Cross-Attention with External Covariates: Inject weather or grid signals.
- Low-Rank Factorization of Dense(128) to trim V2 params while retaining width effect.
- Temporal Mixture-of-Experts: Route seasonal vs. anomaly windows to specialized sub-blocks.
- Calibration Head: Auxiliary network to estimate uncertainty scaling factors.

## 7. Recommendations
- Use Base for rapid iteration & benchmarking consistency.
- Adopt V2 when validation MAE improves ≥3–5% over Base on ≥2 seasons.
- Enable quantile loss only after deterministic convergence to avoid unstable early training.
- Track gate activation distribution; persistent saturation suggests temperature retuning.

## 8. File References
- `Robust_Improved_Hybrid_Model__build_hybrid_model.txt`
- `Robust_Improved_Hybrid_Model_v2__build_hybrid_v2.txt`
- Source notebooks: `Robust_Improved_Hybrid_Model.py`, `Robust_Improved_Hybrid_Model_v2.py`

## 9. Glossary
- Depthwise Separable Conv: Factorized conv reducing parameters vs. standard conv.
- Squeeze‑Excite: Channel-wise attention mechanism scaling feature maps.
- MovingAverage Detrending: Removes low-frequency component to highlight seasonal/residual structure.
- Quantile (Pinball) Loss: Asymmetric loss for estimating conditional quantiles.
- Temperature-Scaled Gate: Divides logits to control sigmoid sharpness.

---
Generated report; refresh after architecture or parameter revisions.
