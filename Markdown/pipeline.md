# BioNet Pipeline (End-to-End)

This document explains the complete workflow of the BioNet project: how data is ingested, models are trained with emissions tracking, metrics and artifacts are saved, predictions are generated from saved weights, and how benchmarking plots and reports are produced.

## 1) Overview

- Goal: Forecast 72-hour renewable generation share (or related signal) per country and season, evaluate models, and track carbon emissions for runs.
- Key components:
  - Data layer: ENTSO‑E live fetch with robust CSV fallback under `Data/`.
  - Model zoo: Converted notebook models in `Models/` (e.g., Transformer, DLinear, Cycle LSTM, Robust Improved Hybrid, etc.).
  - Runner: `scripts/run_with_emissions.py` wraps any entry script with CodeCarbon tracking, image redirection, metrics capture, and weight export.
  - Orchestrator: `main.py` executes selected models across countries and seasons.
  - Predictions: `scripts/predict.py` loads saved weights and produces 72h forecasts plus small reports per model/country/season.
  - Benchmarking & plots: `scripts/benchmark.py` and dedicated plot scripts under `scripts/` aggregate results and render figures.

## 2) Data Ingestion

- Primary source: ENTSO‑E API via `entsoe-py`.
  - Token detection: `.codegreencore.config` or environment variables (`ENTSOE_token`, `ENTSOE_TOKEN`, `CODEGREEN_ENTSOE_TOKEN`, `ENTSOE_API_TOKEN`).
  - Function: `fetch_recent_data_entsoe()` in `scripts/predict.py` (mirrors logic from `scripts/fetch_data.py`).
  - Output: DataFrame with `timestamp` and `renewable_percentage` (or similar target column) for recent `HISTORY_HOURS` (72) hours.
- Fallback: Local CSV under `Data/energy_data_<CC>_5years_<season>.csv` when API is unavailable.
  - Season alignment: controlled by `BIONET_SEASON` or `--season` flags.
- Windowing: `build_input_window()` creates an input tensor with shape `(1, 72, 1)` to feed models; scripts also adapt shapes when SavedModel expects `(None, 72)`.

## 3) Models

- Location: `Models/` contains Python equivalents of the prior notebooks, e.g.:
  - `Transformer_Model.py`, `Transformer_Model_v2.py`, `DLinear_Model.py`, `Cycle_LSTM_Model_v2.py`, `Robust_Improved_Hybrid_Model_v2.py`, `PatchTST_Model_v2.py`, etc.
- Training expectations:
  - Each model file, when executed, performs its own data prep, training, and may produce plots/metrics.
  - The runner captures metrics automatically if the script exposes arrays in globals named like `y_true`/`y_test_inversed` and `y_pred`/`y_pred_inversed`.
- Weight export:
  - The runner searches for a Keras model object in common variables (`model`, `model_v1/v2/v3`, `solar_model`, `best_model`, `forecast_model`) or scans globals for a `tf.keras.Model`.
  - Saved format preference: Keras 3 `export()` → `model.save()` → `tf.saved_model.save()`.

## 4) Training & Emissions Tracking

- Entry: `scripts/run_with_emissions.py`.
  - Redirects `matplotlib.pyplot.savefig` so images save under `Results/images/<Model>/<Country>/<season>/`.
  - In quick mode (`--quick`), monkey‑patches `keras.Model.fit` to cap `epochs` to `--max-epochs` (default 1) for smoke tests.
  - Patches `pandas.read_csv` to enforce seasonal CSV selection when an energy CSV filename is used.
  - Wraps execution with `emissions_tracker()` → CodeCarbon writes CSV logs under `Results/emissions/<Model>/<Country>/<season>/<run>_emissions.csv`.
  - Computes metrics automatically using `scripts/metrics.py` if ground truth and predictions are present in globals.
  - Saves a per‑run JSON report to `Results/reports/<Model>/<Country>/<season>/<run>.json` including metrics, image paths, and link to the emissions CSV.
  - Optionally exports model weights if `SAVE_WEIGHTS=1`.
- Orchestration: `main.py` controls batch execution.
  - Select seasons: `--season summer|winter|both` (defaults to internal `SEASONS` list).
  - Countries: default list includes `DE, DK, ES, HU`; override with `--countries`, `--summer-countries`, `--winter-countries`.
  - Model set: explicit lists at top of `main.py` or auto‑discover all `Models/*.py`; filter with `--filter`.
  - For each model×country×season, runs the model via the runner with env: `BIONET_SEASON`, `ENTRY_MODEL_NAME`, `COUNTRY_CODE`, `WEIGHTS_DIR`, `SAVE_WEIGHTS`.
  - Writes a summary JSON under `Results/reports/seasonal_summary.json` (or seasonal variant) capturing return codes and run names.

## 5) Artifacts & Directory Layout

- `Results/images/<Model>/<Country>/<season>/<files.png>`: Plots saved by training scripts.
- `Results/metrics/<Model>/<Country>/<season>/metrics.json`: Auto‑computed metrics if arrays found.
- `Results/reports/<Model>/<Country>/<season>/<run>.json`: Consolidated per‑run report.
- `Results/emissions/<Model>/<Country>/<season>/<run>_emissions.csv`: CodeCarbon logs.
- `Model Weights/<Model>/<Country>/<season>/<run>/`: SavedModel (or `.keras`/`.h5`) weights.
- `Predictions/<Model>/<Country>/<season>/<run or latest>/`: Forecast JSON, plot, and a compact report per inference.

## 6) Prediction Workflow

- Entry: `scripts/predict.py`.
  - CLI:
    - `--countries "DE,DK,ES,HU"`
    - `--models "Robust_Improved_Hybrid_Model_v2.py,Transformer_Model.py"`
    - `--season summer|winter|both|"summer,winter"`
    - `--run-name` to load a specific weights run folder; defaults to most recent.
    - `--no-fallback` to error instead of using CSV when ENTSO‑E fails.
  - Data acquisition: ENTSO‑E client with CSV fallback; aligns season through `BIONET_SEASON`.
  - Weights loading: Tries `.keras`/`.h5` files or SavedModel directories, using Keras `TFSMLayer` or a signature‑aware wrapper.
  - Input shaping: `build_input_window()` and SavedModel adapter ensure compatibility with both `(None, 72, 1)` and `(None, 72)` signatures.
  - Outputs:
    - Per model/country/season: `forecast_72h.json`, `forecast_plot.png`, `report.{json,txt}` under `Predictions/...`.
    - Aggregated per-season predictions: `Predictions/predictions_<season>.json`.

## 7) Benchmarking & Reporting

- Aggregation: `scripts/benchmark.py` reads `Results/emissions` and `Results/reports` to produce:
  - `Results/Benchmark/emissions_aggregated.csv` and `metrics_aggregated.csv`.
  - Boxplots across models and per‑country.
  - Heatmaps of mean emissions/metrics per model×country×season.
  - Trade‑off scatter plots: metric vs emissions.
  - Top‑N bars by mean metric.
- Additional plots:
  - `scripts/plot_combined_emissions.py`: Combined emissions boxplot (log‑scale) with specified model families, including Cycle LSTM variants.
  - `scripts/plot_emissions_mean_all.py`: Mean emissions per model and multiple distribution plots across all countries/seasons; writes `emissions_stats_all_countries_seasons.csv`.
  - `scripts/plot_mean_mae_by_country.py`: Mean of summer+winter MAE per country, per model.
  - `scripts/plot_dataset_overview.py`: Seasonal time‑series panels, hourly profiles, and a combined Germany summer vs winter plot.

## 8) Configuration & Environment

- Python: `pyproject.toml` requires Python >= 3.9.
- Dependencies: `requirements.txt` includes `numpy, pandas, matplotlib, scikit-learn, torch, codecarbon, codegreen_core, entsoe-py`.
- CodeCarbon: configured via `codecarbon.config` and runtime parameters in `scripts/emissions.py`.
- Key environment vars:
  - `BIONET_SEASON`: season selector for dataset and foldering (`summer`/`winter`).
  - `ENTRY_MODEL_NAME`: used to path emissions and results per model.
  - `COUNTRY_CODE`: two‑letter code used for country folder naming.
  - `WEIGHTS_DIR`: base directory for saving model weights (default `Model Weights`).
  - `SAVE_WEIGHTS`: set `1` to enable weight export in runner.
  - ENTSO‑E token vars: `ENTSOE_token`, `ENTSOE_TOKEN`, `CODEGREEN_ENTSOE_TOKEN`, `ENTSOE_API_TOKEN`.

## 9) Typical Commands (Windows PowerShell)

- Install:

```powershell
pip install -r .\requirements.txt
```

- Run selected models for both seasons with quick mode:

```powershell
python .\main.py --season both --quick --filter Transformer
```

- Run a specific model for Germany (summer), saving weights:

```powershell
$env:COUNTRY_CODE = "DE"; $env:SAVE_WEIGHTS = "1"
python -m scripts.run_with_emissions Models/Robust_Improved_Hybrid_Model_v2.py --run-name Robust_Improved_Hybrid_Model_v2_Germany_summer_run1 --season summer --quick
```

- Generate 72h predictions for both seasons from latest weights:

```powershell
python .\scripts\predict.py --season both --countries "DE,DK,ES,HU" --models "Robust_Improved_Hybrid_Model_v2.py,Transformer_Model.py"
```

- Build benchmark figures and CSVs:

```powershell
python .\scripts\benchmark.py
```

## 10) Notes & Conventions

- Country naming in folders uses human‑readable names (e.g., `Germany`) derived from `COUNTRY_CODE` via `scripts/countries.py`.
- Saved weights may exist as directory SavedModel, `.keras`, or `.h5`; the prediction loader tries these in order.
- Some models export signatures expecting `(None, 72)`; prediction loader adapts shapes accordingly.
- Quick mode is ideal for smoke testing; for real training, omit `--quick` and set appropriate epochs in model scripts.
