# 72‑Hour Renewable Forecast — Plan Greener AI Training (Detailed Slides)

Time: ~10 minutes • Audience: mixed technical/non‑technical • Style: simple, story‑driven

---

## Slide 1 — The Big Idea (hook)

# Introduction

If we can see these patterns 72 hours ahead, we can choose better times to train big models—saving money, reducing emissions, and being kinder to the planet.

That’s the heart of this project: a 72-hour forecast for renewable energy percentage and carbon intensity. Our aim is not just accuracy, but usefulness—simple, reliable insights that tell us “when” is the best time to compute.

What makes this exciting?

- It connects AI with real-world energy trends.
- It helps schedule training when the grid is greenest.
- It opens the door to greener, cheaper, and smarter ML operations.

How do we do it? We use proven time-series models to predict the next 72 hours separated by seasons (winter and summer)

Why does this matter now? Because the need is immediate. AI workloads are growing, and climate goals are urgent. With a clear, beginner-friendly forecast, we can make smarter choices: pause during “dirty” hours, train during “clean” hours.


---

## Slide 2 — Why this is important now? (context)

- AI workloads are growing; energy and emissions matter.
- Renewable supply is highly variable (e.g., Germany, hour‑by‑hour swings).
- Seeing 72 hours ahead = better timing, greener choices.
- Visual: line chart with “clean” and “dirty” hour highlights.

Speaker notes

- Frame the pain: volatile prices, climate goals, budget pressure.

---

## Slide 3 — Project goal (plain language)

- Predict next 72 hours of: renewable energy % and carbon intensity.
- Output: properly evaluated model with low hardware needs.
- Use prediction to schedule compute when the grid is greenest.
- Visual: 72‑hour forecast strip with highs/lows.

Speaker notes

- Keep words simple: “we forecast green hours so we can train then.”

---

## Slide 5 — Dataset overview

This project uses curated energy time‑series for four countries and two seasonal regimes to train, evaluate, and compare 41 forecasting models with carbon accounting enabled.

- Countries covered: Germany, Denmark, Spain, and Hungary
- Time span: 5 years per country (as provided in the CSV filenames)
- Seasonal splits: summer and winter
- Storage layout: `Data/energy_data_<CC>_5years_<season>.csv`
  - Examples:
    - `Data/energy_data_DE_5years_summer.csv`
    - `Data/energy_data_DK_5years_winter.csv`

Each CSV contains a consistent time‑series at a fixed sampling interval with at least a timestamp column and a target signal used for model training. Additional exogenous features (if present) are read directly from the CSV headers and passed to models that support them. The exact columns can be inspected from the CSVs in the `Data/` folder.

### Raw Data Organization

- **Data Files**: 8 CSV files covering 4 countries (DE, DK, ES, HU) with seasonal splits
- **File Pattern**: `energy_data_{COUNTRY_CODE}_{YEARS}years_{season}.csv`
- **Seasons**:
  - Summer: April-September (months 4-9)
  - Winter: October-March (months 10-3)
- **Temporal Resolution**: Hourly data points
- **Time Span**: 5 years of historical data per file
- **Data Size**: ~21,133 hourly records per seasonal dataset (for Germany summer data)

### Data Features

- **Primary Feature**: `renewable_percentage` - the percentage of renewable energy in the grid
- **Index**: `startTime` - timestamp with timezone information (UTC+00:00)
- **Single Feature Focus**: All models use only the renewable percentage as input (N_FEATURES = 1)

## Sequence Creation Parameters

### Time Series Configuration

```python
LOOK_BACK = 72          # Use past 72 hours (3 days) of data to predict
FORECAST_HORIZON = 72   # Predict next 72 hours (3 days ahead)
N_FEATURES = 1          # Single feature: renewable_percentage
```

### Sequence Generation Process

The `create_sequences()` function transforms the time series data into supervised learning format:

1. **Input Sequences (X)**: 72 consecutive hourly values representing 3 days of historical renewable energy percentages
2. **Target Sequences (y)**: 72 consecutive future hourly values representing 3 days of forecasted renewable energy percentages
3. **Sliding Window**: Creates overlapping sequences with a step size of 1 hour

## Train-Validation-Test Split Strategy

### Split Ratios (Consistent Across All Models)

- **Training Set**: 70% of sequences
- **Validation Set**: 15% of sequences
- **Test Set**: 15% of sequences

---

## Slide 4 — The models (simple, diverse)

- Classics: LSTM/CycleLSTM, DLinear, N‑BEATS.
- Transformers family: Transformer, TFT, Autoformer, Informer, PatchTST.
- Hybrids: CNN‑LSTM, Hybrid CNN–CycleLSTM + Attention, EnsembleCI, Mamba.
- Visual: grid of model cards (v1/v2/v3).

Speaker notes

- One line per family: “LSTMs remember sequences”, “Transformers focus on important time parts”.

---

## Slide 5 — The Best models

##### The DLinear model (Decomposition-Linear) is a time-series forecasting model. It’s a simple but powerful baseline model.

### 3. Architectural Details

### 3.1 Base DLinear Model

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

## 4. Layer-by-Layer Snapshots

(See exported summaries for full detail.)

| Variant | Key Layers (Sequential Summary)                                                     |
| ------- | ----------------------------------------------------------------------------------- |
| Base    | Dense(72 seasonal), Dense(72 trend), Add                                            |
| V2      | MovingAverage, Subtract, Dense(72 seasonal), Dense(72 trend), Add                   |
| V3      | MovingAverage, Subtract, Dense(512), Dropout, Dense(72) (×2 parallel branches), Add |

# Transformer Model

### 4.1 Base Model

Framework:

- Input: `(72, 1)` univariate window.
- Encoder Blocks: Attention (MultiHeadAttention with key_dim ~1 inferred) → Dropout → Residual → LayerNorm → 2-layer feed-forward (Dense(4) → Dropout → Dense(1)) → Residual. Pattern repeated.
- Pooling: `GlobalAveragePooling1D` over (time, channel_first config in notebook; summary shows channels_last on some variants).
- Head: Dense(64) → Dropout → Dense(72 output horizon).
- Missing positional embedding: relies solely on model capacity to infer temporal ordering—suboptimal for permutation-sensitive tasks.
- Strength: Extremely lightweight (≈13K params) enabling very fast inference.
- Weakness: Limited representation power; no explicit temporal position encoding may degrade longer-horizon pattern retention.

## 4. Layer-by-Layer Snapshots

##### The Transformer model is a deep learning architecture introduced in the paper. It’s based entirely on the attention mechanism.

| Variant | Key Layers (Sequential Summary)                                                                                                                                                                       |
| ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Base    | Input → (MHA → Dropout → Residual → LN → Dense(4) → Dropout → Dense(1) → Residual) ×2 → GlobalAvgPool1D → Dense(64) → Dropout → Dense(72)                                                             |
| V2      | Input → Token+PosEmbedding → [ (LN → MHA(2 heads, 256) → Dropout → Residual → LN → Dense(4) → Dropout → Dense(256) → Residual) ×2 ] → GlobalAvgPool1D → Dense(64) → Dropout → Dense(72)               |
| V3      | Input → Token+PosEmbedding → [ (LN → MHA(4 heads, 256) → Dropout → Residual → LN → Dense(4) → Dropout → Dense(256) → Residual) ×4 ] → GlobalAvgPool1D → Dense(256) → Dropout → Dense(128) → Dense(72) |

# The Robust Improved Hybrid Model
The Robust Improved Hybrid family fuses multiple inductive biases for 72‑hour renewable energy forecasting:

- Local pattern extraction (temporal convolutions / depthwise separable convs in V2)
- Sequence memory (BiLSTM, GRU fusion in V2)
- Global dependency modeling (stacked Multi-Head Attention blocks)
- Linear decomposition (DLinear-style seasonal + trend heads; detrending enhanced in V2)
- Adaptive gating (learned sigmoid gate; temperature scaling + squeeze‑excite in V2)

Goal: Balance interpretability (linear decomposition) with expressiveness (deep contextualized latent) while controlling overfitting through architectural regularization.


## 3.a Layer-by-Layer Snapshots
| Variant | Key Layers (Sequential Summary) |
|---------|---------------------------------|
| Base | Input → Conv1D(32) → LN → (MHA→Residual) ×2 → Conv1D(64) → Conv1D(64) → BiLSTM(64) → BiLSTM(32) → GlobalAvgPool (attention path) || Linear Path: Flatten → Dense(72 seasonal) + Dense(72 trend) → Add || Gating: Dense→Sigmoid blend |
| V2 | Input → PosEncoding → (DepthwiseConv→PointwiseConv→Residual) ×2 → (PreNorm: LN→MHA(32 key_dim, width 128)→Residual→LN→FFN(small)→Residual) ×2 → Squeeze-Excite → Split: Recurrent (BiLSTM(64)+GRU(128) add) & Attention pooled → Concat → Dense(128) → Dropout → Dense(72) || Linear Path: MA(window=7) trend + Residual seasonal → (Dense(72 trend) + Dense(72 seasonal with Dropout)) → Add || Gate: Dense / temperature-scaled Sigmoid |

# Add Carbon Emission plots
# xx
---

## Slide 4 — What we built (Pipeline)


Diagram hobe

Speaker notes

- Mention standardized outputs: images, metrics, emissions, summaries.

---

## Slide 8 — What we save (so you can trust it)

- Plots: forecasts per model/country/season.
- Metrics: MAE, MSE, RMSE, R² (long‑horizon aware).
- Emissions: per‑run CSV via CodeCarbon.
- Visual: folder tree for Results/images, metrics, emissions.

Speaker notes

- Stress traceability: every run leaves plots, numbers, and emissions records.

---

## Slide 9 — Measuring impact (CodeCarbon)

- Tracks energy use and emissions per run.
- Scoped by model/country/season for fair comparison.
- On Windows: power estimation works; sensors (Intel Power Gadget) improve accuracy.
- Visual: small CO₂ badge per run.

Speaker notes

- “We measure so we can improve; we compare apples to apples.”

---

## Slide 10 — Sample outcome (Germany)

- Summer and winter pipelines run end‑to‑end.
- Artifacts saved: plots, metrics.json, emissions.csv, summaries.
- Use these to spot greener windows and plan training.
- Visual: side‑by‑side summer vs winter snapshots.

Speaker notes

- Share a quick real metric (e.g., MAE/ RMSE) and an example green window.

---

## Slide 11 — What you get (benefits)

- A “weather report” for clean energy hours.
- Actionable scheduling for heavy training jobs.
- Lower cost, lower emissions, same model quality.
- Visual: calendar with training blocks on green hours.

Speaker notes

- Tie benefits to teams: MLOps, researchers, finance, sustainability.

---

## Slide 12 — Limits (honest and practical)

- Windows power estimation may be noisy; sensors recommended.
- Quick mode/timeout trades speed for depth.
- Minimal HPO; baseline focus is workflow + viability.
- Visual: caution icons with short notes.

Speaker notes

- “We’re transparent. These are solvable with small next steps.”

---

## Slide 13 — Roadmap (next steps)

- Exact‑match model selection; early stopping; mixed precision.
- Carbon‑intensity‑aware scheduling signals.
- Data caching & versioning; energy benchmark suite; accelerator profiling.
- Visual: 3‑step path with milestones.

Speaker notes

- Invite contributions; small changes, real gains.

---

## Slide 14 — Call to action

- Use the forecast to plan one training window this week.
- Start with one country; extend to more as needed.
- Measure every run with CodeCarbon; share the results.
- Visual: “Start → Measure → Share” loop.

Speaker notes

- Make the ask concrete and low‑friction.

---

## Slide 15 — Backup: how to run (for Q&A)

- Summer: `python main_summer.py --countries DE --quick`
- Winter: `python main_winter.py --countries DE --quick`
- Add: `--filter Transformer` and `--timeout 300` as needed.
- Visual: command with labels (country, filter, quick).

Speaker notes

- Keep this as a reference slide for questions.

---

## Slide 16 — Backup: where files go

- Results/images/<Model>/<Country>/<Season>/
- Results/metrics/<Model>/<Country>/<Season>/metrics.json
- Results/reports/<model|season summaries>.json
- Results/emissions/<Model>/<Country>/<Season>/<run>\_emissions.csv

Speaker notes

- Reinforce transparency and repeatability.

---

## Slide 17 — Backup: model families (41 variants)

- 14 families × versions (mostly v1/v2/v3; one family has v1/v2).
- Coverage: LSTM/CycleLSTM, DLinear, N‑BEATS, Transformers (Transformer, TFT, Autoformer, Informer, PatchTST), CNN‑LSTM, Hybrid, EnsembleCI, Mamba, Robust Improved Hybrid.
- See `ModelDescription.md` for a short summary per family.

Speaker notes

- Variety enables robust comparisons across regions and seasons.

---

## Slide 18 — Backup: glossary (plain terms)

- Renewable %: share of electricity coming from renewables.
- Carbon intensity: grams of CO₂ per kWh.
- Green hours: higher renewable %, lower carbon intensity.
- Quick mode: fewer epochs; faster results; lower compute.

Speaker notes

- Use if the audience is new to energy terms.

---

## Slide 19 — Backup: risks & mitigations

- Measurement noise → add sensors, cross‑check on Linux.
- Data gaps → API fallback, caching.
- Heavy models → early stopping, mixed precision, timeouts.
- Adoption → keep CLI simple, share summaries.

Speaker notes

- Show a pragmatic path to production readiness.

---

## Slide 20 — Thank you

- One line: “Let’s train when the wind is on our side.”
- Contact / repo link (optional).

Speaker notes

- Invite a quick first pilot: one model, one country, one week.
