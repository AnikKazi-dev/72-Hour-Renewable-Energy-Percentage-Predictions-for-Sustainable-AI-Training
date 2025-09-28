## Benchmark report

This report consolidates accuracy and carbon-emissions measurements for 41 model variants, evaluated across 4 countries and 2 seasons with CodeCarbon tracking. Sources are the aggregated CSVs and plots in `Results/Benchmark`.

### Executive summary

- Accuracy leaders by slice are stable and consistent with the Top‑10 charts (see details below). DLinear/Cycle‑LSTM/Robust Hybrid families dominate the best MAE slots depending on country/season.
- Emissions are highly skewed: most trainings are ≤0.10 kg CO₂e, while a few Transformer v3 runs reach ~0.6–1.0 kg CO₂e per training across countries.
- Accuracy vs. emissions shows weak coupling (Pearson corr(MAE, emissions_kg) ≈ +0.112). Low‑carbon models can be both accurate and efficient.
- Practical picks: Denmark (Robust_Improved_Hybrid_v2), Germany (Cycle_LSTM_v2), Hungary (DLinear / Robust_Improved_Hybrid), Spain (DLinear_v2). Avoid Transformer_Model_v3 for routine runs due to high emissions and inconsistent accuracy.

### Inputs analyzed

- `Results/Benchmark/metrics_aggregated.csv` — 329 rows with columns: model, country, season, run, mae, rmse, mse, r2, mape, emissions_csv, report_path.
- `Results/Benchmark/emissions_aggregated.csv` — emissions_kg, energy_kwh, duration_sec per run.
- Plots: boxplots, heatmaps, trade‑off scatter, and per‑country+season Top‑10 MAE charts (e.g., `top10_mae_Denmark_summer.png`).

### Coverage snapshot

- Models: 41 distinct
- Countries: 4 (Germany, Denmark, Spain, Hungary)
- Seasons: 2 (summer, winter)
- Runs in metrics: 329
- Joinable emissions per run: available for the same set of runs

Note: Germany‑winter shows N=42 rows (one more than the expected 41), indicating a duplicate or an extra run variant. Everything else has N≈41 per country‑season.

### Performance summary (MAE)

Best and worst models per country‑season, with mean MAE across all models for context:

- Denmark
  - Summer: best Robust_Improved_Hybrid_Model_v2 (MAE 5.053), worst N_Beats_Model_v3 (MAE 9.060), mean 6.133, N=41
  - Winter: best Transformer_Model (MAE 9.515), worst Transformer_Model_v3 (MAE 12.258), mean 10.134, N=41
- Germany
  - Summer: best Cycle_LSTM_Model_v2 (MAE 9.433), worst Transformer_Model_v3 (MAE 14.624), mean 11.089, N=41
  - Winter: best Cycle_LSTM_Model (MAE 11.640), worst Autoformer_Model (MAE 14.591), mean 12.650, N=42
- Hungary
  - Summer: best DLinear_Model (MAE 3.979), worst Mamba_Model (MAE 20.824), mean 8.606, N=41
  - Winter: best Robust_Improved_Hybrid_Model (MAE 4.514), worst Mamba_Model_v3 (MAE 9.840), mean 6.230, N=41
- Spain
  - Summer: best DLinear_Model_v2 (MAE 4.162), worst Transformer_Model_v3 (MAE 10.632), mean 5.910, N=41
  - Winter: best DLinear_Model (MAE 6.422), worst Mamba_Model_v2 (MAE 9.454), mean 7.527, N=41

See also:

- `boxplot_mae_all_models.png`, `boxplot_rmse_all_models.png` — overall distributions
- `heatmap_mean_mae_summer.png`, `heatmap_mean_mae_winter.png` — mean MAE by model×country
- `top10_mae_<Country>_<season>.png` — country+season leaderboards

### Emissions and trade‑offs

- Weak association between accuracy and emissions: Pearson corr(MAE, emissions_kg) ≈ +0.112 across joined runs (i.e., lower emissions don’t systematically harm MAE).
- Emissions distributions are heavy‑tailed. Most models cluster below 0.10 kg per training. Transformer_Model_v3 exhibits the highest per‑run emissions in Denmark, Germany, Hungary, and Spain (≈0.6–1.0 kg range in the per‑country boxplots).
- DLinear variants, CNN‑LSTM family, and several hybrid models are among the lowest‑emission performers while remaining competitive on MAE.
- Reference plots: `tradeoff_mae_vs_emissions.png`, `tradeoff_rmse_vs_emissions.png`.
- Distributions and seasonality: `boxplot_emissions_all_models.png` plus per‑country boxplots `boxplot_emissions_<Country>.png`.

#### Pareto front and outliers (from trade‑off scatter)

- The scatter (`tradeoff_mae_vs_emissions.png`) shows a dense, overlapping cluster for all countries at ≤0.10 kg CO₂e spanning nearly the full MAE range. This confirms there is no inherent need to emit more to be accurate.
- A few high‑emission outliers (~0.55–1.10 kg) appear with mid‑pack accuracy (MAE roughly 7–12). These points are dominated by multiple low‑emission runs that match or beat their MAE—i.e., they are not Pareto‑efficient.
- Practical policy: set a soft cap at 0.10 kg CO₂e per training. Promote models only if they beat the best low‑carbon baselines by a meaningful margin (e.g., ≥0.2–0.3 MAE) on the target slice.
- Likely Pareto‑efficient families at low emissions: DLinear (v1/v2), CNN‑LSTM, Robust_Improved_Hybrid, and Cycle‑LSTM. Conversely, Transformer_Model_v3 produces the most carbon‑intensive points without clear accuracy benefits and is usually dominated.

### Seasonal insights

- Denmark, Germany: winter is harder (higher mean MAE) than summer.
- Spain: winter degrades vs summer (mean MAE 7.527 vs 5.910).
- Hungary: winter improves vs summer (mean MAE 6.230 vs 8.606), suggesting stronger winter signal/predictability or better model fit.

### Model family notes

- DLinear family performs strongly in Spain (summer and winter leaders).
- Cycle‑LSTM variants lead in Germany; Transformers (v3) tend to underperform in summer across multiple countries.
- Mamba variants show instability, taking worst in several slices (Hungary summer/winter, Spain winter).
- Robust_Improved_Hybrid variants are competitive in Denmark (summer best) and Hungary (winter best).

### Heatmap highlights (emissions • MAE • RMSE)

- Emissions (summer/winter): `Transformer_Model_v3` is the most carbon‑intensive across all four countries; Denmark‑summer shows the strongest outlier band. DLinear, CNN‑LSTM, and several hybrids stay in the lightest bands (lowest emissions) for both seasons.
- MAE: Germany tends darker than other countries in summer; Hungary winter tends lighter (lower MAE) across many families. Informer/Mamba variants show darker bands in several slices, in line with the Top‑10 MAE charts.
- RMSE mirrors MAE patterns: Cycle‑LSTM and DLinear families are consistently among the lighter bands (better), while Transformer v3 and some Mamba/Informer variants are darker in multiple slices.
- Seasonal deltas: The relative ranking of models is fairly stable between summer and winter; the same families remain low‑carbon and competitive on error metrics.

### Recommendations (balanced accuracy × emissions)

- Prefer these defaults for routine training:
  - Denmark: `Robust_Improved_Hybrid_Model_v2`
  - Germany: `Cycle_LSTM_Model_v2`
  - Hungary: `DLinear_Model` or `Robust_Improved_Hybrid_Model`
  - Spain: `DLinear_Model_v2`
- Low‑carbon baselines: DLinear family (v1/v2) and CNN‑LSTM family have consistently low emissions in the boxplots.
- Avoid for cost‑sensitive runs: `Transformer_Model_v3` (order‑of‑magnitude higher emissions without commensurate accuracy gains).

### Artifacts index (selected)

- Aggregates: `metrics_aggregated.csv`, `emissions_aggregated.csv`
- Accuracy: `boxplot_mae_all_models.png`, `heatmap_mean_mae_summer.png`, `heatmap_mean_mae_winter.png`
- Emissions: `boxplot_emissions_all_models.png`, `heatmap_mean_emissions_summer.png`, `heatmap_mean_emissions_winter.png`
- Trade‑offs: `tradeoff_mae_vs_emissions.png`, `tradeoff_rmse_vs_emissions.png`
- Country+season leaderboards: `top10_mae_<Country>_<season>.png`

### How to reproduce (Windows PowerShell)

1. Generate runs (CodeCarbon tracked). Use `main.py` to orchestrate seasons/countries/models. Examples:

```powershell
# Run both seasons for specific countries and repeat each training twice
python .\main.py --season both --countries DE,DK,ES,HU --repeat 2

# Narrow to a model family (substring match) and quick smoke test
python .\main.py --season summer --filter DLinear --quick --countries DE,DK,ES,HU
```

2. Aggregate results and build plots:

```powershell
python -m scripts.benchmark
```

Outputs will be written under `Results/Benchmark/` (e.g., `metrics_aggregated.csv`, `emissions_aggregated.csv`, and all plots referenced here).

### Data quality and next steps

- Investigate Germany‑winter (N=42) for duplicates or extra variants; align to 41 where appropriate.
- Consider reporting per‑run emissions summaries (mean/median) by model family to contextualize accuracy vs cost.
- Optionally include confidence intervals or bootstrap CIs on mean MAE per slice to quantify uncertainty.

Additional checks to consider:

- Compute Pareto‑efficient fronts (MAE vs emissions) per country/season to highlight “no‑regret” models.
- Report energy (kWh) and duration (sec) alongside emissions for the same runs to separate hardware/region effects.
- Track versioned configs for each model run (epochs, batch size, seeds) in `Results/reports/.../run.json` to improve reproducibility.
