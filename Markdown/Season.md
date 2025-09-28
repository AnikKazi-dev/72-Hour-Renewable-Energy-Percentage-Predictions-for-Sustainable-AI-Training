## Dataset overview

This project uses curated energy time‑series for four countries and two seasonal regimes to train, evaluate, and compare 41 forecasting models with carbon accounting enabled.

- Countries covered: Germany, Denmark, Spain, and Hungary
- Time span: 5 years per country (as provided in the CSV filenames)
- Seasonal splits: summer and winter
- Storage layout: `Data/energy_data_<CC>_5years_<season>.csv`
  - Examples:
    - `Data/energy_data_DE_5years_summer.csv`
    - `Data/energy_data_DK_5years_winter.csv`

Each CSV contains a consistent time‑series at a fixed sampling interval with at least a timestamp column and a target signal used for model training. Additional exogenous features (if present) are read directly from the CSV headers and passed to models that support them. The exact columns can be inspected from the CSVs in the `Data/` folder.

## Why summer and winter are split

Energy systems behave differently across seasons. Splitting the data by season improves model fit, benchmarking fidelity, and interpretability.

- Distinct demand patterns: cooling vs. heating loads dominate at different times of year.
- Generation mix shifts: PV output peaks in summer; wind often dominates in winter, changing signal dynamics.
- Daylight and temperature effects: daylight length and temperature distributions shift autocorrelations and seasonality.
- Stationarity: training a single model on mixed regimes can violate stationarity assumptions; season‑specific training reduces distribution drift.
- Cleaner benchmarking: per‑season metrics make comparisons fairer (e.g., a model strong on PV‑driven summers vs. wind‑driven winters).

## How seasons are used in this project

- Training and results: The runner sets a `season` context, and artifacts are saved under `Results/<...>/<season>/` with emissions, images, and reports separated per season.
- Model weights: Saved under `Model Weights/<Model>/<Country>/<season>/<run>/` so summer and winter checkpoints don’t conflict.
- Prediction: `scripts/predict.py` loads the appropriate seasonal weights and writes outputs under `Predictions/<...>/<season>/`.
- Benchmarking: `scripts/benchmark.py` aggregates metrics from all JSON reports and plots per‑season comparisons (e.g., `heatmap_mean_mae_summer.png`, `heatmap_mean_mae_winter.png`) and country+season leaderboards.

## Countries and files in Data/

- Germany: `energy_data_DE_5years_summer.csv`, `energy_data_DE_5years_winter.csv`
- Denmark: `energy_data_DK_5years_summer.csv`, `energy_data_DK_5years_winter.csv`
- Spain: `energy_data_ES_5years_summer.csv`, `energy_data_ES_5years_winter.csv`
- Hungary: `energy_data_HU_5years_summer.csv`, `energy_data_HU_5years_winter.csv`

If you add more countries, follow the same naming convention (`<ISO2>` code and `<season>` suffix) to plug into the existing pipeline without changes.

## Row counts per CSV

Counts are data rows (header excluded):

- Germany:
  - `energy_data_DE_5years_summer.csv`: 21,132 rows
  - `energy_data_DE_5years_winter.csv`: 19,658 rows
- Denmark:
  - `energy_data_DK_5years_summer.csv`: 21,132 rows
  - `energy_data_DK_5years_winter.csv`: 19,658 rows
- Spain:
  - `energy_data_ES_5years_summer.csv`: 21,132 rows
  - `energy_data_ES_5years_winter.csv`: 19,658 rows
- Hungary:
  - `energy_data_HU_5years_summer.csv`: 21,132 rows
  - `energy_data_HU_5years_winter.csv`: 19,658 rows

## Timeframes per CSV

Timestamps are in UTC (+00:00):

- Germany:
  - summer (`energy_data_DE_5years_summer.csv`): 2021-04-01 00:00 to 2025-08-27 11:00 UTC
  - winter (`energy_data_DE_5years_winter.csv`): 2020-12-31 22:00 to 2025-03-31 23:00 UTC
- Denmark:
  - summer (`energy_data_DK_5years_summer.csv`): 2021-04-01 00:00 to 2025-08-27 11:00 UTC
  - winter (`energy_data_DK_5years_winter.csv`): 2020-12-31 22:00 to 2025-03-31 23:00 UTC
- Spain:
  - summer (`energy_data_ES_5years_summer.csv`): 2021-04-01 00:00 to 2025-08-27 11:00 UTC
  - winter (`energy_data_ES_5years_winter.csv`): 2020-12-31 22:00 to 2025-03-31 23:00 UTC
- Hungary:
  - summer (`energy_data_HU_5years_summer.csv`): 2021-04-01 00:00 to 2025-08-27 11:00 UTC
  - winter (`energy_data_HU_5years_winter.csv`): 2020-12-31 22:00 to 2025-03-31 23:00 UTC

## Sampling interval between rows

- Primary interval: 60 minutes (hourly) across all datasets during their respective seasonal periods.
- Non-uniform gaps: Large jumps of ~262,140–263,580 minutes (≈182–183 days) occur between seasonal blocks across years because only summer or only winter months are included in each file. This is expected and does not indicate missing hourly data within a season.
