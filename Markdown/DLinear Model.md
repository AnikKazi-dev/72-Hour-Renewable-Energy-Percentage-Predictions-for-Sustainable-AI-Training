# DLinear Model Family Report

## 1. Overview
The DLinear family implements a decomposition-based multi-step forecasting approach. It separates the input sequence into trend and seasonal components and learns direct mappings to a fixed forecast horizon (72 hours). This design prioritizes simplicity, interpretability, and low computational overhead (for Base/V2), while V3 explores increased representational capacity via non-linear heads.

## 2. Variant Summary
| Variant | Params | Core Additions | Decomposition Strategy | Regularization |
|---------|--------|----------------|------------------------|----------------|
| Base    | 10,512 | Two parallel linear projections | Direct seasonal + trend Dense layers | None explicit |
| V2      | 10,512 | MovingAverage trend smoother | Seasonal = Input − Smoothed Trend | Implicit (structural) |
| V3      | 148,624 | MLP (512→Dropout→72) per branch + L2 | Same as V2 (MA + residual seasonal) | Dropout + L2 |

## 3. Architectural Details
### 3.1 Base
Framework:
- Input: 72-length univariate vector
- Parallel Dense(seasonal, 72), Dense(trend, 72)
- Add(seasonal, trend) → Output (72)
Characteristics:
- Pure linear mapping per component
- Minimal parameters → extremely fast inference
- Susceptible to overfitting only in very noisy regimes (already small)

### 3.2 V2 (Structural Regularization)
Framework Additions:
- MovingAverage kernel (size≈inferred from code: default 25 in V3; V2 code implies similar) applied to input → trend_input
- Seasonal component = input − trend_input
- Two linear heads: Dense(trend, 72), Dense(seasonal, 72) → Add
Effects:
- Smooths trend → reduces noise leakage
- Maintains identical parameter count (no added trainable weights in smoothing layer)
- Provides more interpretable decomposition under seasonal fluctuation scenarios

### 3.3 V3 (Non-linear Expansion)
Framework Additions:
- Trend branch: MovingAverage → Dense(512, ReLU, L2=0.001) → Dropout(0.2) → Dense(72)
- Seasonal branch: Residual (input − trend_input) → Dense(512, ReLU, L2=0.001) → Dropout(0.2) → Dense(72)
- Fusion: Add(trend_out, seasonal_out)
Effects:
- Introduces non-linear feature extraction while retaining decomposition structure
- Param growth: 14.1× relative to Base/V2
- Improved capacity for capturing subtle regime shifts or non-linear seasonal modulation
- Risk: Higher overfitting potential; relies on Dropout + L2 to stabilize

## 4. Layer-by-Layer Snapshots
(See exported summaries for full detail.)

| Variant | Key Layers (Sequential Summary) |
|---------|---------------------------------|
| Base | Dense(72 seasonal), Dense(72 trend), Add |
| V2 | MovingAverage, Subtract, Dense(72 seasonal), Dense(72 trend), Add |
| V3 | MovingAverage, Subtract, Dense(512), Dropout, Dense(72) (×2 parallel branches), Add |

## 5. Comparative Analysis
### 5.1 Capacity vs. Simplicity
- Base / V2: Ultra-light; suitable baseline and strong when data volume is modest.
- V3: Adds representational depth; better for complex, multi-regime temporal patterns.

### 5.2 Interpretability
- Base: Highest; linear seasonal + trend easily attributable.
- V2: Preserves interpretability while improving trend isolation (smoother).
- V3: Partially reduced interpretability (MLP transformations obscure direct coefficient meaning) but decomposition boundary still clear.

### 5.3 Risk / Overfitting
- Base/V2: Low risk; may underfit complex signals.
- V3: Medium risk; mitigated with L2 + Dropout. Needs validation monitoring.

### 5.4 Computational Cost
| Variant | Relative Inference Cost | Comment |
|---------|-------------------------|---------|
| Base    | 1× (reference)          | Single matrix ops |
| V2      | 1.02×                   | Extra moving average pass (O(n)) |
| V3      | ~5–7×                   | Two 512-unit MLP branches + dropout |

## 6. When to Use Which
| Scenario | Recommended Variant | Rationale |
|----------|---------------------|-----------|
| Fast baseline / deployment on edge | Base | Lowest latency & clear decomposition |
| Need smoother trend isolation | V2 | MovingAverage stabilizes trend estimate |
| Complex seasonal distortions / non-linear effects | V3 | MLP heads capture richer interactions |
| Limited data / high noise | V2 | Structural smoothing without added params |
| Strong regularization budgets & accuracy priority | V3 | Higher ceiling with regularization |

## 7. Potential Improvements
- Adaptive Kernel Size: Learnable or data-driven moving average window.
- Branch Attention: Weight seasonal vs. trend contributions dynamically.
- Multi-Resolution Decomposition: Add intermediate-scale component (e.g., wavelet or dilated average).
- Quantile Head Extension: Multi-output distributional forecasting (P10, P50, P90).
- Lightweight Non-Linearity (V2.5): Replace 512-unit MLP with low-rank factorization to balance capacity and size.

## 8. Recommendations
- Keep Base as a reproducible reference in benchmarks.
- Promote V2 as default production choice: stability + interpretability.
- Use V3 selectively: trigger when validation improvements > X% over V2 across ≥2 seasons.
- Add automated early stopping & learning-rate schedule for V3 (already partially present) plus monitoring of parameter norm growth.

## 9. File References
- `DLinear_Model__build_dlinear_model.txt`
- `DLinear_Model_v2__build_dlinear_model.txt`
- `DLinear_Model_v3__build_dlinear_model.txt`

## 10. Quick Glossary
- MovingAverage Layer: Non-trainable smoothing via AvgPool1D.
- Decomposition: Separation of raw input into trend + seasonal pathways.
- L2 Regularization: Penalizes large weights to reduce overfitting.

---
Generated on demand. Update by re-running export + refreshing this report if architecture files change.
