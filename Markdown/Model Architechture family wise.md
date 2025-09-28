# Model Architectures Overview

This document summarizes the architectural design of each model family and its variants used in the project. For full layer-by-layer details (shapes, parameters, connections), refer to the generated `.txt` summaries in this same folder (one per model) and the fallback `.png` diagrams.

---
## 1. Autoformer Family (Auto-Correlation Based Transformer)
Files: `Autoformer_Model__build_autoformer_model.txt`, `Autoformer_Model_v2__build_autoformer_model.txt`, `Autoformer_Model_v3__build_autoformer_model.txt`

Core Ideas:
- Leverages Auto-Correlation mechanism instead of standard attention to capture long-term periodic dependencies.
- Encoder-style stacked blocks with: LayerNorm → AutoCorrelation → Residual → LayerNorm → Conv1D Feed-Forward (1x1 projection style) → Residual.
- Progressive temporal abstraction over fixed input window (72 timesteps).
- Global average pooling aggregates encoded temporal features before final regression head.

High-Level Structure:
1. Input (sequence length 72, feature dim 1)
2. Linear embedding (Dense to latent width, e.g. 64)
3. Repeated Encoder Blocks (2–3+ depending on variant):
   - LayerNorm
   - AutoCorrelation (triplet form inputs inside block)
   - Dropout + Residual Add
   - LayerNorm
   - Conv1D (bottleneck down to small channel count, e.g. 4)
   - Dropout
   - Conv1D (expand back to latent width)
   - Residual Add
4. GlobalAveragePooling1D (channel aggregation)
5. MLP Head: Dense → Dropout → Dense (forecast horizon = 72)

Differences by Variant:
- v2/v3 likely increase depth or latent width (check param counts in their `.txt`).
- Parameter scaling primarily via latent dimension and number of encoder blocks.

---
## 2. CarbonCast Family (Temporal Convolution + Attention Hybrid)
Files: `CarbonCast_Model__build_carboncast_model.txt` (+ v2/v3 variants)

Concept:
- Hybrid temporal feature extractor combining dilated / causal Conv1D layers with attention-style or gating enhancements (see builder parameters in code if present).
- Emphasizes efficient receptive field growth with moderate parameter count.

Typical Components:
- Input → Stacked Conv1D (causal or same padding)
- Normalization (BatchNorm or LayerNorm depending on variant)
- Optional residual merges
- Global pooling (avg) or flatten
- Dense projection layers to horizon

Variant Scaling:
- v2/v3 increase channel width or add blocks.
- Potential added dropout / regularization depth.

---
## 3. CNN-LSTM Family
Files: `CNN_LSTM_Model__build_cnnlstm_model.txt`, `..._v2`, `..._v3`

Pipeline:
1. Input (72×1)
2. Temporal Convolutions:
   - Conv1D (128 filters, kernel 3, causal) → BatchNorm → Dropout
   - Conv1D (64 filters, kernel 3, causal) → BatchNorm → Dropout
3. Sequence Modeling:
   - LSTM (256 units)
4. Deep Projection Head:
   - Dense (256) → BatchNorm → Dropout
5. Output Layer:
   - Dense (72) (multi-step direct forecast)

Variant Evolution:
- v2: May adjust filter counts / dropout.
- v3: May add deeper convolutional stack or increased recurrent width.

Strengths:
- CNN front-end accelerates pattern extraction.
- LSTM captures cross-hour seasonal/diurnal dependencies.

---
## 4. Cycle LSTM Family
Files: `Cycle_LSTM_Model__build_cycle_lstm_model.txt` (+ v2/v3)

Design:
- Multiple LSTM passes / cyclical fusion aiming to reinforce temporal encoding.
- Potential twin LSTMs whose outputs are merged (Add) to stabilize gradients.
- Downstream global pooling and dense forecast head.

Block Pattern:
- Input → (Optional light Conv or embedding) → Parallel/Sequential LSTMs → Add/Merge → Attention/Gating (if present) → Pool → Dense(s) → Output.

Variants scale with number of recurrent layers and hidden size.

---
## 5. DLinear Model
Files: `DLinear_Model__build_dlinear_model.txt` (+ v2/v3)

Concept:
- Decomposition-based linear forecasting (Trend + Seasonal separation) simplified into channel-wise linear projections.
- Extremely parameter-efficient; relies on linear mappings over the look-back window.

Structure:
- Input (72×1)
- Optional decomposition pre-processing (implicit inside builder)
- Linear (Dense / 1D projection) layers producing horizon directly
- Minimal or no deep nonlinear stack.

Use Case:
- Baseline efficiency reference vs complex deep models.

---
## 6. EnsembleCI Model (Ensemble with Confidence / Interaction)
Files: `EnsembleCI_Model__build_ensemble_model.txt` (+ v2/v3)

Idea:
- Aggregates predictions from multiple lightweight internal learners or transformed views (channels, seasonality slices, etc.).
- May include channel interaction (CI) fusion layer.

Generic Flow:
1. Input projection / splitting
2. Parallel lightweight transformations
3. Concatenation / Add fusion
4. Dense aggregation → Horizon output

Variants add parallel branches or increase fusion dimensionality.

---
## 7. Hybrid CNN + Cycle LSTM + Attention Model
Files: `Hybrid_CNN_CycleLSTM_Attention_Model__build_hybrid_model.txt` (+ v2/v3)

Key Components:
1. Conv1D Front-End: Temporal feature lifting (e.g., 64 channels) + BatchNorm + Dropout.
2. Dual LSTM Streams: Two LSTMs (same hidden size) process sequence; outputs combined via Add (ensemble stabilization).
3. Channel Attention / Re-weighting:
   - Permute → Dense (channel recalibration) → Permute back.
   - Multiply with residual LSTM features.
4. GlobalAveragePooling1D (feature condensation).
5. Dense → Dropout → Dense (forecast horizon = 72).

Benefits:
- CNN local pattern capture, LSTM temporal memory, lightweight attention-style channel gating.
- Balanced parameter footprint vs depth.

Variant Differences:
- v2/v3 expand hidden sizes or attention gating dimensionality.

---
## 8. Informer Family (Efficient Transformer Variant)
Files: `Informer_Model__build_informer_model.txt` (+ v2/v3)

Characteristics:
- Sparse attention concept (Informer style) approximated via reduced channel MultiHeadAttention + Conv1D feedforward.
- Encoder stack with (LayerNorm → MHA → Residual → LayerNorm → Conv1D (expansion) → Dropout → Conv1D (projection) → Residual).
- Final global average pooling + MLP head.

Simplified Block:
- LN → MHA → Dropout → Residual → LN → Conv1D (small width) → Dropout → Conv1D (restore) → Residual.

Variants adjust number of encoder layers and latent width.

---
## 9. Mamba Model Family
Files: `Mamba_Model__build_mamba_model.txt` (+ v2/v3)

Concept:
- (Assumed) Sequence state-space or selective scan inspired architecture (naming aligned with Mamba SSM family).
- Likely uses custom sequential mixing blocks instead of classical attention (check builder code for exact ops).

General Layout:
- Input embedding
- Repeated state-space or gated mixing blocks (Normalization → Mixing → Feedforward → Residual)
- Global pooling
- Dense forecast head

Variant scaling via block count and hidden width.

---
## 10. N-BEATS Family
Files: `N_Beats_Model__build_nbeats_model.txt` (+ v2/v3)

Essentials:
- Backcast/Forecast fully connected stacks with residual trend & seasonality basis decomposition.
- Repetitive blocks: (FC → ReLU)*k → Split into backcast & forecast projection.

Conceptual Pipeline:
1. Input flatten / embedding
2. Stacked N-BEATS Blocks (trend + seasonal bases)
3. Summed forecast outputs across blocks

Variants add block depth or internal width.

---
## 11. PatchTST Family (Patch-based Transformer)
Files: `PatchTST_Model__build_patchtst_model.txt` (+ v2/v3)

Key Ideas:
- Splits time sequence into overlapping/contiguous patches; embeds patches as tokens.
- Transformer encoder over patch tokens (MHA + FFN) with positional encodings.
- Patch re-assembly or projection to multi-step forecast.

High-Level Flow:
1. Patch Extraction & Linear Projection
2. Positional / Learnable Embedding
3. Repeated Transformer Blocks
4. Pool / Flatten
5. Dense Output (Horizon = 72)

Variants scale patch dimension, number of heads, encoder depth.

---
## 12. Robust Improved Hybrid Models
Files: `Robust_Improved_Hybrid_Model__build_hybrid_model.txt`, `..._v2__build_hybrid_v2.txt`

Architecture Theme:
- Enhanced fusion of multiple temporal encoders (e.g., CNN + LSTM + attention gating + residual linear path) with robustness tweaks (extra normalization / dropout / gated scaling).
- Emphasis on resilience to noise and season shifts.

Flow (Abstract):
1. Multi-branch temporal feature extraction
2. Attention / gating fusion
3. Global pooling
4. Dense refinement
5. Output layer

v2 introduces structural or gating refinements (see param increase in `.txt`).

---
## 13. Temporal Fusion Transformer (TFT) Family
Files: `Temporal_Fusion_Transformer_Model__build_tft_model.txt` (+ v2/v3)

Fundamentals:
- LSTM encoder backbone + gating & multi-head attention integration.
- Variable selection / gating (approximated through dense + multiply blocks).
- Multi-block temporal fusion before pooling and MLP head.

Component Breakdown (Observed):
1. Input → Dense embedding → LSTM (temporal context)
2. Residual Add with embedding
3. LayerNorm → Gated Residual Network (Dense → Dense + Multiply path)
4. MultiHeadAttention + Residual + LayerNorm (temporal fusion)
5. Repeat GRN + Attention stack
6. GlobalAveragePooling1D
7. Dense (hidden) → Dropout → Dense (forecast)

Variant Scaling:
- v2 increases attention head parameters.
- v3 increases hidden dimensionality / stacked depth.

---
## 14. Transformer Family
Files: `Transformer_Model__build_transformer_model.txt` (+ v2/v3)

Baseline Transformer (Small):
- Repeated minimal encoder blocks with (LayerNorm → MultiHeadAttention → Dropout → Residual → LayerNorm → FeedForward (Dense→Dropout→Dense) → Residual).
- GlobalAveragePooling1D aggregates time dimension.
- Dense (hidden) → Dropout → Dense output (72 steps).

Variants:
- v2 adds token + positional embedding module, larger hidden width (256), deeper stack.
- v3 dramatically scales embedding (1024) and depth (multi-stack expansions) for high-capacity modeling.

---
## 15. Patch-Based / Other Minor Variants
(If additional bespoke experimental models exist, they follow hybridization of: embedding → temporal core (Conv/LSTM/Attention/SSM) → pooling → MLP head.)

---
## Cross-Cutting Design Notes
- Input Window: Consistently 72 past timesteps predicting 72-future horizon (direct multi-output regression).
- Normalization Layers: LayerNormalization for transformer-style blocks; BatchNormalization in CNN-heavy stacks.
- Pooling Strategy: GlobalAveragePooling1D used to compress temporal dimension before dense heads in attention-based and hybrid models.
- Regularization: Dropout widely applied (0.1–0.4 range depending on model depth).
- Output Head: Almost all models end with one or two Dense layers mapping latent representation to 72-length forecast vector.

---
## Quick Reference Table (Conceptual)
| Family | Core Mechanism | Sequence Modeling | Pooling | Forecast Head |
|--------|----------------|-------------------|---------|---------------|
| Autoformer | Auto-Correlation Blocks | Encoded periodicity | Global Avg | Dense MLP |
| CarbonCast | Conv + Hybrid Attention | Dilated/causal conv | Global Avg | Dense |
| CNN-LSTM | Conv Front + LSTM | LSTM (256) | N/A (flatten via LSTM) | Dense |
| Cycle LSTM | Multiple LSTM cycles | Dual/merged LSTMs | Global Avg / Flatten | Dense |
| DLinear | Linear decomposition | None (linear proj) | N/A | Dense |
| EnsembleCI | Parallel learners fusion | Simple transforms | Concat/Add | Dense |
| Hybrid CNN-CycleLSTM-Attn | CNN + Dual LSTM + Channel Attention | LSTM + gating | Global Avg | Dense |
| Informer | Sparse-like MHA + Conv FFN | Stacked encoder | Global Avg | Dense |
| Mamba | State-space mixing (assumed) | SSM blocks | Global Avg | Dense |
| N-BEATS | Backcast/Forecast Blocks | Fully-connected stacks | Summation | Dense proj |
| PatchTST | Patch token Transformer | MHA Encoder | Pool/Flatten | Dense |
| Robust Hybrid | Multi-branch fusion | Mixed (CNN/LSTM/Attn) | Global Avg | Dense |
| TFT | LSTM + GRN + MHA | LSTM + Attention | Global Avg | Dense |
| Transformer | Standard Encoder Blocks | MHA + FFN | Global Avg | Dense |

---
## How To Get Full Detail
For any model: open the matching `.txt` (e.g., `Transformer_Model_v2__build_transformer_model.txt`) to see every layer name, output shape, and parameter count.

---
## Reference Style Inspiration
Structure formatting follows the concise hierarchical pattern similar to the provided scCDCG example (module-oriented breakdown with bullet hierarchy).

---
*End of Document*
