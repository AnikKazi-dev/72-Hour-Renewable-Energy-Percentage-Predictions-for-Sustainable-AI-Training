<!-- Variant-separated scCDCG-style documentation -->
# Deep Learning Forecast Model Architectures (41 Variants)

Each variant now has an independent section (no grouped bullets) matching the style of your scCDCG example: Heading, optional subtitle/context, then a Framework block with hierarchical bullets (● primary, ○ sub, ▹ detail). Param counts come from the exported summaries. Open the referenced `.txt` files for full layer listings; PNGs are fallback textual diagrams.

Legend: ● primary  |  ○ sub  |  ▹ detail  |  ≈ inferred  |  Δ change vs. base

---

## Autoformer Base
Autoformer: Auto-Correlation Transformer for long-horizon forecasting.
Framework:
● Embedding & Input
  ○ Input window: 72×1 → Dense embedding (~64)
● Encoder Blocks (Repeated)
  ○ LayerNorm → AutoCorrelation op → Residual add
  ○ LayerNorm → Conv1D bottleneck ↓ → Dropout → Conv1D restore ↑ → Residual
● Output Head
  ○ GlobalAveragePooling1D → Dense → Dropout → Dense(72)
Parameters: 18.83K  |  Summary: `Autoformer_Model__build_autoformer_model.txt`

## Autoformer V2
Δ Reduced latent width / depth for efficiency.
Framework:
● Same macro stages as Base with narrower internal channels.
Parameters: 15.25K  |  Summary: `Autoformer_Model_v2__build_autoformer_model.txt`

## Autoformer V3
Δ Tuned regularization / internal width (same param count as V2).
Framework:
● Mirrors V2 structure; adjustments are training-oriented (≈ dropout/width tweaks).
Parameters: 15.25K  |  Summary: `Autoformer_Model_v3__build_autoformer_model.txt`

## CarbonCast Base
Hybrid temporal conv + attention/gating.
Framework:
● Convolutional Stack: Temporal Conv1D layers (expanding receptive field)
● Fusion/Gating: Normalization + gating/attention combination
● Head: Global pooling → Dense(72)
Parameters: 125.13K  |  Summary: `CarbonCast_Model__build_carboncast_model.txt`

## CarbonCast V2
Δ Wider channels + extra fusion block.
Framework:
● Deeper/wider conv stack; enhanced fusion depth.
Parameters: 185.55K  |  Summary: `CarbonCast_Model_v2__build_carboncast_model.txt`

## CarbonCast V3
Δ Large capacity scale (multi-stage widening & depth).
Framework:
● Expanded conv/fusion depth; larger dense head.
Parameters: 700.62K  |  Summary: `CarbonCast_Model_v3__build_carboncast_model.txt`

## CNN-LSTM Base
Convolutional front-end feeding recurrent temporal modeling.
Framework:
● Local Feature Extraction: Conv1D(128) → BN → Dropout → Conv1D(64) → BN → Dropout
● Sequence Modeling: LSTM(256)
● Projection: Dense(256) → BN → Dropout → Dense(72)
Parameters: 439.94K  |  Summary: `CNN_LSTM_Model__build_cnnlstm_model.txt`

## CNN-LSTM V2
Δ Reduced widths for parameter efficiency.
Framework:
● Slimmed conv filters / LSTM units.
Parameters: 211.36K  |  Summary: `CNN_LSTM_Model_v2__build_cnnlstm_model.txt`

## CNN-LSTM V3
Δ Expanded widths for higher expressiveness.
Framework:
● Wider conv layers and larger LSTM / dense layers.
Parameters: 801.10K  |  Summary: `CNN_LSTM_Model_v3__build_cnnlstm_model.txt`

## Cycle LSTM Base
Reinforced temporal encoding via cyclical/parallel LSTM passes.
Framework:
● Multiple LSTM paths → Add fusion → Dense head
Parameters: 158.92K  |  Summary: `Cycle_LSTM_Model__build_cycle_lstm_model.txt`

## Cycle LSTM V2
Δ Minor parameter reduction (regularization focus).
Framework:
● Slightly narrower LSTM paths.
Parameters: 152.78K  |  Summary: `Cycle_LSTM_Model_v2__build_cycle_lstm_model.txt`

## Cycle LSTM V3
Δ Larger hidden sizes & deeper fusion.
Framework:
● Expanded LSTM capacities before fusion.
Parameters: 570.06K  |  Summary: `Cycle_LSTM_Model_v3__build_cycle_lstm_model.txt`

## DLinear Base
Linear seasonal + trend decomposition with additive fusion.
Framework:
● Decomposition: Parallel Dense(trend) + Dense(seasonal)
● Fusion: Add → Forecast vector (72)
Parameters: 10.51K  |  Summary: `DLinear_Model__build_dlinear_model.txt`

## DLinear V2
Δ Added MovingAverage smoothing for trend isolation.
Framework:
● Trend: MovingAverage → Linear projection
● Seasonal: Residual (input - trend) → Linear projection → Add
Parameters: 10.51K  |  Summary: `DLinear_Model_v2__build_dlinear_model.txt`

## DLinear V3
Δ Non-linear MLP heads + L2 regularization for both branches.
Framework:
● Trend: MovingAverage → Dense(512, ReLU, L2) → Dropout → Dense(72)
● Seasonal: Residual → Dense(512, ReLU, L2) → Dropout → Dense(72)
● Fusion: Add(trend, seasonal)
Parameters: 148.62K  |  Summary: `DLinear_Model_v3__build_dlinear_model.txt`

## EnsembleCI Base
Parallel learners with interaction fusion.
Framework:
● Branch Ensemble: Multiple light learners (conv / dense / recurrent)
● Fusion: Add/Concat → Dense aggregation
● Output: Dense(72)
Parameters: 208.07K  |  Summary: `EnsembleCI_Model__build_ensemble_model.txt`

## EnsembleCI V2
Δ Additional branches and wider fusion.
Framework:
● Expanded branch set + wider aggregation layer.
Parameters: 330.96K  |  Summary: `EnsembleCI_Model_v2__build_ensemble_model.txt`

## EnsembleCI V3
Δ Large-scale ensemble expansion.
Framework:
● Many widened branches → Large fusion head.
Parameters: 1.36M  |  Summary: `EnsembleCI_Model_v3__build_ensemble_model.txt`

## Hybrid CNN-CycleLSTM-Attention Base
Hybrid local + recurrent + channel attention.
Framework:
● CNN Front: Conv1D stack
● Dual LSTMs: Parallel temporal encoding → Add
● Channel Recalibration: Dense over permuted channels → Multiply
● Head: GAP → Dense → Dropout → Dense(72)
Parameters: 229.20K  |  Summary: `Hybrid_CNN_CycleLSTM_Attention_Model__build_hybrid_model.txt`

## Hybrid CNN-CycleLSTM-Attention V2
Δ Lean variant for efficiency.
Framework:
● Reduced widths in conv/LSTM layers.
Parameters: 150.66K  |  Summary: `Hybrid_CNN_CycleLSTM_Attention_Model_v2__build_hybrid_model.txt`

## Hybrid CNN-CycleLSTM-Attention V3
Δ Larger LSTMs + expanded attention.
Framework:
● Wider recurrent channels and attention projection.
Parameters: 604.49K  |  Summary: `Hybrid_CNN_CycleLSTM_Attention_Model_v3__build_hybrid_model.txt`

## Informer Base
Sparse-ish attention + convolutional feed-forward.
Framework:
● Encoder Block: LN → MHA(reduced) → Residual → LN → Conv1D bottleneck ↓ → Dropout → Conv1D ↑ → Residual
● Stacking: Repeated blocks → GAP → Dense(72)
Parameters: 8.43K  |  Summary: `Informer_Model__build_informer_model.txt`

## Informer V2
Δ Increased latent dimensions / depth.
Framework:
● More blocks & wider attention heads.
Parameters: 573.07K  |  Summary: `Informer_Model_v2__build_informer_model.txt`

## Informer V3
Δ Regularization/struct tweaks (same size as V2).
Framework:
● Architecture parity with V2.
Parameters: 573.07K  |  Summary: `Informer_Model_v3__build_informer_model.txt`

## Mamba Base
State-space inspired selective sequence mixing.
Framework:
● Embedding → Repeated mixing blocks → Pool → Dense(72)
Parameters: 246.66K  |  Summary: `Mamba_Model__build_mamba_model.txt`

## Mamba V2
Δ Compressed capacity.
Framework:
● Fewer / narrower mixing blocks.
Parameters: 43.27K  |  Summary: `Mamba_Model_v2__build_mamba_model.txt`

## Mamba V3
Δ Expanded hidden sizes and/or extra blocks.
Framework:
● More / wider mixing layers.
Parameters: 347.02K  |  Summary: `Mamba_Model_v3__build_mamba_model.txt`

## N-BEATS Base
Backcast/forecast block stacks (generic interpretable architecture).
Framework:
● Stacked Fully Connected Blocks (FC→ReLU×k)
● Dual Heads: Backcast & Forecast → Additive forecast accumulation
Parameters: 1.15M  |  Summary: `N_Beats_Model__build_nbeats_model.txt`

## N-BEATS V2
Δ Reduced depth/width.
Framework:
● Fewer / narrower blocks.
Parameters: 331.44K  |  Summary: `N_Beats_Model_v2__build_nbeats_model.txt`

## N-BEATS V3
Δ Deeper & wider configuration.
Framework:
● More blocks or larger hidden units.
Parameters: 1.54M  |  Summary: `N_Beats_Model_v3__build_nbeats_model.txt`

## PatchTST Base
Patch tokenization + Transformer encoding.
Framework:
● Patch Extraction: Sequence → patch tokens
● Token Embedding: Linear + positional
● Encoder: Transformer layers → Pool/Flatten → Dense(72)
Parameters: 8.43K  |  Summary: `PatchTST_Model__build_patchtst_model.txt`

## PatchTST V2
Δ Increased embedding / heads / depth.
Framework:
● Wider embedding & additional encoder layers.
Parameters: 147.98K  |  Summary: `PatchTST_Model_v2__build_patchtst_model.txt`

## PatchTST V3
Δ Structural adjustments under same budget.
Framework:
● Similar parameter envelope; redistributed internal widths.
Parameters: 147.98K  |  Summary: `PatchTST_Model_v3__build_patchtst_model.txt`

## Robust Improved Hybrid Base
Multi-branch temporal fusion with robustness enhancements.
Framework:
● Branches: CNN + LSTM + linear/others
● Fusion: Gating / combination → Pool → Dense(72)
Parameters: 203.68K  |  Summary: `Robust_Improved_Hybrid_Model__build_hybrid_model.txt`

## Robust Improved Hybrid V2
Δ Added robustness (extra gating / wider layers).
Framework:
● Expanded branch widths and gating depth.
Parameters: 306.74K  |  Summary: `Robust_Improved_Hybrid_Model_v2__build_hybrid_v2.txt`

## Temporal Fusion Transformer Base
LSTM backbone with gated residual networks + attention fusion.
Framework:
● Embedding → LSTM backbone
● Gated Residual Networks (feature processing)
● MultiHeadAttention fusion cycles + residual/LN
● Head: GAP → Dense → Dropout → Dense(72)
Parameters: 454.73K  |  Summary: `Temporal_Fusion_Transformer_Model__build_tft_model.txt`

## Temporal Fusion Transformer V2
Δ Slimmed architecture for efficiency.
Framework:
● Reduced hidden sizes / attention dimensions.
Parameters: 150.66K  |  Summary: `Temporal_Fusion_Transformer_Model_v2__build_tft_model.txt`

## Temporal Fusion Transformer V3
Δ Expanded hidden & attention dimensions.
Framework:
● Larger GRNs and attention heads.
Parameters: 603.46K  |  Summary: `Temporal_Fusion_Transformer_Model_v3__build_tft_model.txt`

## Transformer Base
Standard encoder-style architecture for time-series.
Framework:
● Encoder Blocks: (LN → MHA → Dropout → Residual → LN → FFN(Dense→Dropout→Dense) → Residual) × depth
● Head: GAP → Dense(64) → Dropout → Dense(72)
Parameters: 12.97K  |  Summary: `Transformer_Model__build_transformer_model.txt`

## Transformer V2
Δ Deeper & wider with explicit token/positional embeddings.
Framework:
● More encoder blocks + larger embedding dims.
Parameters: 573.07K  |  Summary: `Transformer_Model_v2__build_transformer_model.txt`

## Transformer V3
Δ Very wide initial embedding (1024) then staged reductions.
Framework:
● Wide early representation → progressive narrowing across blocks.
Parameters: 561.30K  |  Summary: `Transformer_Model_v3__build_transformer_model.txt`

---
All 41 model variants exported with individual framework sections (fallback PNGs due to missing Graphviz).

End of document.