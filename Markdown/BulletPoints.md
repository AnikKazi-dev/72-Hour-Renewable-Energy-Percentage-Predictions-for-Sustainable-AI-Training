- Abstract

  - Goal: season- and country-aware renewable forecasting; emissions-measured
  - Contributions: script-only pipeline; unified runner; CodeCarbon integration
  - Artifacts: centralized images/metrics/reports/emissions; per-run summaries

- Keywords

  - Renewable forecasting; time series; deep learning; sustainability; CodeCarbon; Europe; seasonality

- Introduction

  - Motivation: grid decarbonization; carbon-aware ML practice
  - Problem: forecast renewable_percentage across seasons/countries
  - Approach: multi-model benchmarking with emissions accounting

- Related work (brief)

  - Transformers/TFT/Autoformer; DLinear/N-BEATS; green ML measurement (CodeCarbon)

- Data

  - Source: codegreen_core; entsoe-py fallback
  - Horizon: last 5 years; hourly cadence
  - Coverage: EU countries; dynamic COUNTRY_CODE
  - Splits: train/val/test; rolling windows; seasonal subsets (summer/winter)
  - Files: energy*data*{CC}_5years_{season}.csv
  - Preprocessing: TZ-safe alignment; normalization; season filters

- Methodology

  - Orchestration
    - Entrypoints: main_summer.py, main_winter.py
    - Args: --filter, --countries, --quick, --timeout, --repeat, --extra
    - Centralized Results/ hierarchy; per-model/per-country/per-season
  - Models
    - Transformers, TFT (v1–v3), Autoformer, Informer
    - CNN-LSTM, Cycle LSTM, DLinear, N-BEATS, Mamba, EnsembleCI, Robust Hybrid
    - Standardized I/O; env-driven COUNTRY_CODE; season-aware loaders
  - Training
    - Quick mode (epoch cap); headless plotting
    - Timeouts for heavy runs; batch continuity
  - Emissions accounting
    - CodeCarbon per-run CSV; model/country/season scoping
    - Windows CPU estimation; Intel Power Gadget note
  - Evaluation
    - Metrics: MAE, MSE, RMSE, R²
    - Artifacts: forecast plots; metrics.json; season summaries
  - Reproducibility
    - Env vars for season/country/run; consistent layout; JSON reports

- Experimental setup

  - OS: Windows; Python; TensorFlow/Keras; pandas/numpy/sklearn; Matplotlib (Agg)
  - Power measurement: CPU estimation; optional Intel Power Gadget
  - Runtime configs: quick mode; timeouts (smoke runs)

- Results

  - Smoke tests: Germany (summer/winter)
  - Verified outputs: images; metrics.json; emissions.csv; summaries

- Discussion

  - Trade-offs: speed vs accuracy (quick/timeout)
  - Granularity: country/season analysis; carbon-aware insights
  - Standardization: comparability; automation; maintainability

- Limitations

  - Windows power estimation bias
  - Truncated training under timeouts
  - Minimal HPO; limited tests

- Future work

  - Exact-match selection; early stopping; mixed precision
  - Carbon-intensity-aware scheduling
  - Caching and data versioning; model registry
  - Unit tests; energy benchmark suite
  - Sensor integration; accelerator profiling

- Conclusion

  - Emissions-measured, season/country-aware forecasting pipeline
  - Reproducible results; centralized artifacts
  - Foundation for green ML benchmarking across regions/seasons

- Ethics & sustainability

  - Measurement-first; transparency; efficiency practices
  - Responsible compute; reproducibility

- Availability
  - CLI entrypoints; season/country datasets; Results/ hierarchy
