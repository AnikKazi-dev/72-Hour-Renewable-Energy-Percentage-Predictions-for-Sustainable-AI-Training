# Model Comparison

This document summarizes and compares all model families present in `Models/` (14 families, 3 versions each where applicable, totaling 41 variants). For each family we cover: what it is, why it fits this project, key design choices, input-output contract, strengths/limitations, and differences across v1/v2/v3.

Note: Names map 1:1 to files in `Models/`. Each script has been standardized to train on renewable percentage time series (hourly), use 72h context (window) and 72h horizon, and to be season/country aware.

## Common setup

- Input: univariate sequence of renewable_percentage (%), window=72 (shape [B, 72, 1])
- Output: 72-step forecast (%), shape [B, 72]
- Metrics: MAE, RMSE, R² (computed post inverse transform)
- Training: Keras/TensorFlow with quick mode and emissions tracking via the runner
- Data: ENTSO-E per-country; seasonal CSV fallback

---

## 1) Autoformer (Autoformer_Model[,_v2,_v3].py)

- What: Decomposition-Transformer for long-term series forecasting with autocorrelation-based attention; seasonal-trend decomposition inside the network.
- Why here: Strong baseline for longer horizons; handles periodicity in energy/renewables.
- Strengths: Captures seasonal patterns; efficient via auto-correlation mechanism; good for 72h horizon.
- Limitations: Heavier than linear baselines; sensitive to scaling and sequence length.
- Versions:
  - v1: Baseline widths/heads; vanilla Autoformer block stack.
  - v2: Deeper encoder or longer kernel; dropout tuned for generalization.
  - v3: Wider hidden dims + improved positional embedding.

## 2) CarbonCast (CarbonCast_Model[,_v2,_v3].py)

- What: Project variant focused on carbon/renewables forecasting; typically hybrid feature projection + temporal backbone.
- Why: Tailored for energy decarbonization signals; provides interpretability hooks.
- Strengths: Pragmatic architecture; balances bias/variance on our data.
- Limitations: Not a published canonical model; more engineering than theory.
- Versions: progressive refinements in feature embedding, regularization, and head.

## 3) CNN-LSTM (CNN_LSTM_Model[,_v2,_v3].py)

- What: 1D conv feature extractors followed by LSTM; common sequence baseline.
- Why: Robust on noisy univariate series; fast to train; strong short-to-mid horizon.
- Strengths: Good inductive bias (local patterns + temporal memory).
- Limitations: Can underperform on very long seasonal cycles without explicit seasonality modules.
- Versions: more filters/depth; residual connections; tuned dropout.

## 4) Cycle-LSTM (Cycle_LSTM_Model[,_v2,_v3].py)

- What: LSTM with explicit cyclical skip/residual connections to reinforce daily/weekly cycles.
- Why: Renewable % exhibits daily cycles; this helps for 72h.
- Strengths: Simple, stable training; decent performance-cost tradeoff.
- Limitations: LSTM capacity limits very long-term structure.
- Versions: increasing hidden size, skip topology tweaks, layer norm.

## 5) DLinear (DLinear_Model[,_v2,_v3].py)

- What: Decomposition-Linear model: separate linear heads for trend/seasonal; minimalistic yet strong baseline.
- Why: Extremely fast and surprisingly competitive on long horizons.
- Strengths: Low compute; good generalization; interpretability.
- Limitations: Limited nonlinearity; may miss complex regime shifts.
- Versions: multi-head trend/seasonal, window size tuning, per-country scaling.

## 6) EnsembleCI (EnsembleCI_Model[,_v2,_v3].py)

- What: Light ensemble over complementary backbones (e.g., linear + recurrent + attention) with calibration.
- Why: Smooths idiosyncrasies; often boosts stability across countries/seasons.
- Strengths: Robust aggregate; reduces variance; better worst-case.
- Limitations: Heavier to train/save; debugging harder.
- Versions: expand member set; improve weights via validation; regularize ensemble head.

## 7) Hybrid CNN + CycleLSTM + Attention (Hybrid_CNN_CycleLSTM_Attention_Model[,_v2,_v3].py)

- What: Hybrid feature pyramid (CNN) + recurrent backbone (CycleLSTM) + attention readout.
- Why: Combines local pattern extraction with temporal memory and global focus.
- Strengths: Versatile; strong on mixed regimes.
- Limitations: More hyperparameters; careful tuning needed.
- Versions: deeper CNN, refined attention (scaled dot-product), dropout.

## 8) Informer (Informer_Model[,_v2,_v3].py)

- What: Sparse self-attention Transformer for long sequences with ProbSparse attention.
- Why: Efficient long-range modeling for horizon 72 while keeping compute in check.
- Strengths: Handles long contexts; memory efficient vs vanilla transformer.
- Limitations: More complex training dynamics; hyperparam sensitive.
- Versions: head count/hidden size/decoder depth tweaks; look-back window tuning.

## 9) Mamba (Mamba_Model[,_v2,_v3].py)

- What: State Space Model (SSM) backbone (Mamba-like) for sequence modeling.
- Why: Promising SSM efficiency; handles long-range with linear-time complexity.
- Strengths: Scales well; good for long contexts; strong inductive bias for sequences.
- Limitations: Newer stack; implementation maturity varies.
- Versions: kernel length/state dims tuned; residual gating; stabilized training.

## 10) N-BEATS (N_Beats_Model[,_v2,_v3].py)

- What: Backcast/Forecast fully-connected stacks with basis expansions.
- Why: Strong univariate baseline; interpretable trend/seasonal blocks.
- Strengths: Competitive on many datasets; transparent components.
- Limitations: MLP-heavy; can be parameter-hungry.
- Versions: stack depths, block types, basis choices refined per version.

## 11) PatchTST (PatchTST_Model[,_v2,_v3].py)

- What: Transformer on patchified time series; ViT-style for sequences.
- Why: Strong recent results on long-horizon forecasting.
- Strengths: Captures local and global structure; good sample efficiency.
- Limitations: Patch sizing sensitive; needs careful normalization.
- Versions: patch length/stride tuning; positional encoding variants.

## 12) Robust Improved Hybrid (Robust_Improved_Hybrid_Model[,_v2].py)

- What: Upgraded hybrid stack built to avoid popup/interactive artifacts and stabilize training.
- Why: Practical high-robustness option; good on difficult countries/seasons.
- Strengths: Defensive preprocessing; strong regularization; non-interactive plotting fixed.
- Limitations: Larger runtime; more components.
- Versions: v2 tightens regularization, improves scheduler/early-stopping.

## 13) Temporal Fusion Transformer (Temporal_Fusion_Transformer_Model[,_v2,_v3].py)

- What: TFT with gating, variable selection, and interpretable attention.
- Why: Rich interpretability for planners; handles static/time-varying covariates (used in simplified form here).
- Strengths: Strong for multivariate; useful attention insights.
- Limitations: Heavier and complex; on univariate may be overkill but still solid.
- Versions: progressively lighter heads for univariate; tuned dropout and gating.

## 14) Transformer (Transformer_Model[,_v2,_v3].py)

- What: Baseline encoder-decoder Transformer adapted to univariate forecasting.
- Why: Standard attention baseline for comparison.
- Strengths: Flexible; well-understood.
- Limitations: Quadratic attention cost; can overfit with small data.
- Versions: width/depth/regularization increments across v2/v3.

---

## Cross-family comparison (high level)

- Speed vs accuracy:
  - Fast: DLinear, CNN-LSTM, Cycle-LSTM
  - Balanced: Autoformer, PatchTST, N-BEATS
  - Heavy but powerful: Informer, TFT, Hybrid stacks, Ensembles
  - Experimental/efficient long-range: Mamba
- Robustness (noisy regimes): Hybrid, EnsembleCI, CNN-LSTM, Cycle-LSTM
- Long-horizon structure: Autoformer, Informer, PatchTST, Mamba, N-BEATS
- Interpretability: N-BEATS (basis), TFT (attention/gating), DLinear (decomp)

## Why these models for this project

- Renewable % has daily/weekly seasonality and occasional shocks; we need a spectrum:
  - Linear/decomposition baselines to set a floor and provide interpretability.
  - Recurrent CNN/LSTM variants for local-to-mid-term patterns.
  - Attention/Transformer variants for global dependencies.
  - SSM (Mamba) for efficient long contexts.
  - Hybrid/Ensemble for difficult regimes and stability.
- All integrate cleanly into the training/prediction pipeline (72→72), save Keras 3 compatible weights, and generate standardized reports and plots.

## Version deltas (pattern)

- v1: canonical/baseline settings taken from references or standard configs.
- v2: regularization improvements, mild capacity increase, tuned learning rate/epochs.
- v3: architecture width/depth or advanced positional modules; sometimes longer window.

## Practical guidance

- Quick baseline: DLinear, CNN-LSTM, Cycle-LSTM.
- Highest accuracy for long horizons: Autoformer, PatchTST, Informer; try Mamba.
- Tough/noisy countries: Hybrid_CNN_CycleLSTM_Attention, EnsembleCI, Robust_Improved_Hybrid.
- Need interpretability: N-BEATS, TFT, DLinear.

## Notes on training and artifacts

- Weights are saved under: `Model Weights/<Model>/<Country Full Name>/<season>/<run_name>`.
- Predictions mirror the same structure in `Predictions/` and include JSON, plot, and a brief report.
- CodeCarbon tracks emissions under `Results/emissions/<Model>/<Country Full Name>/<season>/` with per-run CSVs.
