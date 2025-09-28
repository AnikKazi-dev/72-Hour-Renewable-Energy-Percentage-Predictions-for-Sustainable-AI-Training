# Benchmark.py Workflow Documentation

## Overview

The `benchmark.py` script provides comprehensive analysis and visualization of model performance across multiple dimensions including accuracy metrics, carbon emissions, and computational efficiency.

## Workflow Stages

### 1. Data Source Identification

- **Input Directories:**
  - `Results/emissions/<Model>/<Country>/<Season>/` - Contains CSV files with emissions data
  - `Results/reports/<Model>/<Country>/<Season>/` - Contains JSON files with metrics data

### 2. Data Collection Phase

#### Emissions Data Collection (`collect_emissions()`)

- Scans all subdirectories under `Results/emissions/`
- Identifies `*_emissions.csv` files
- Extracts context from file paths: model, country, season, run name
- Calls `_load_emissions_total()` for each CSV file
- Aggregates: emissions (kg CO₂e), energy consumption (kWh), duration (seconds)

#### Metrics Data Collection (`collect_metrics_from_reports()`)

- Scans all subdirectories under `Results/reports/`
- Identifies `*.json` report files
- Extracts performance metrics: MAE, RMSE, MSE, R², MAPE
- Links to corresponding emissions data files

### 3. Data Processing and Validation

#### Path Parsing (`_parse_ctx_from_path()`)

- Validates directory structure
- Extracts metadata: model name, country, season, run identifier
- Handles path structure variations

#### Emissions Processing (`_load_emissions_total()`)

- Reads CodeCarbon CSV output
- Handles cumulative vs. incremental data formats
- Extracts total emissions, energy, and duration values
- Manages missing or malformed data

### 4. Data Aggregation

#### DataFrame Creation

- **Emissions DataFrame (dfe):** Model, Country, Season, Run, Emissions_kg, Energy_kwh, Duration_sec
- **Metrics DataFrame (dfm):** Model, Country, Season, Run, MAE, RMSE, MSE, R², MAPE
- **Combined DataFrame (dfme):** Merged emissions and metrics data

#### CSV Export

- `save_aggregated()` → `emissions_aggregated.csv`
- `save_metrics_aggregated()` → `metrics_aggregated.csv`

### 5. Visualization Generation

#### Emissions Analysis

1. **Box Plots (`plot_box_all_models()`)**

   - Distribution of emissions across all 40+ model variants
   - Jitter points colored by country
   - Focus on specific countries for clarity

2. **Heatmaps (`plot_heatmaps()`)**

   - Model × Country emissions matrix
   - Separate heatmaps for summer/winter seasons
   - Mean emissions aggregation

3. **Country-Specific Analysis (`plot_per_country_box()`)**
   - Individual box plots for focus countries (Denmark, Germany, Hungary, Spain)
   - Model comparison within each country

#### Metrics Analysis

1. **Performance Box Plots (`plot_metric_box_all_models()`)**

   - MAE and RMSE distributions across models
   - Identifies best and worst performing models

2. **Metric Heatmaps (`plot_metric_heatmaps()`)**

   - Performance matrices by season
   - Model × Country accuracy comparison

3. **Top Models Identification (`plot_top_models_bar()`)**
   - Horizontal bar charts of best performing models
   - Ranked by mean MAE/RMSE (lower is better)

#### Advanced Analysis

1. **Trade-off Analysis (`plot_tradeoff_scatter()`)**

   - Scatter plots: Performance vs. Emissions
   - Identifies efficient models (low emissions, high accuracy)
   - Country-specific color coding

2. **Country-Season Specific Analysis**
   - `plot_top10_mae_per_country_from_csv()`: Top 10 models per country/season
   - `plot_histogram_mae_per_country_from_csv()`: MAE distributions by country/season

### 6. Output Organization

#### File Structure

```
Results/Benchmark/
├── emissions_aggregated.csv
├── metrics_aggregated.csv
├── boxplot_emissions_all_models.png
├── heatmap_mean_emissions_summer.png
├── heatmap_mean_emissions_winter.png
├── boxplot_emissions_<country>.png
├── boxplot_mae_all_models.png
├── boxplot_rmse_all_models.png
├── heatmap_mean_mae_summer.png
├── heatmap_mean_mae_winter.png
├── heatmap_mean_rmse_summer.png
├── heatmap_mean_rmse_winter.png
├── tradeoff_mae_vs_emissions.png
├── tradeoff_rmse_vs_emissions.png
├── top10_mae.png
├── top10_rmse.png
├── top10_mae_<country>_<season>.png
└── histogram_mae_<country>_<season>.png
```

### 7. Key Features

#### Statistical Analysis

- **Central Tendency:** Mean, median calculations across runs
- **Distribution Analysis:** Box plots show quartiles and outliers
- **Correlation Analysis:** Performance vs. efficiency trade-offs
- **Ranking Systems:** Model performance hierarchies

#### Visualization Capabilities

- **Multi-dimensional Comparison:** Model × Country × Season × Metric
- **Focus Countries:** Detailed analysis for key European markets
- **Seasonal Patterns:** Summer vs. winter performance differences
- **Efficiency Metrics:** Carbon footprint vs. accuracy trade-offs

#### Robustness Features

- **Error Handling:** Graceful handling of missing or malformed data
- **Flexible Input:** Supports various CSV and JSON formats
- **Scalability:** Handles 40+ models across 25+ countries
- **Extensibility:** Easy addition of new metrics or visualization types

## Usage Examples

### Basic Benchmark Generation

```bash
python scripts/benchmark.py
```

### Integration with Training Pipeline

```bash
# After training models with main.py
python main.py --season both --countries DE,DK,ES,HU
python scripts/benchmark.py  # Generates comprehensive analysis
```

## Dependencies

- **Core:** pandas, numpy, matplotlib
- **Optional:** seaborn (enhanced visualizations)
- **Data Sources:** CodeCarbon emissions tracking, model training results

## Output Interpretation

### Performance Metrics

- **MAE (Mean Absolute Error):** Lower values indicate better accuracy
- **RMSE (Root Mean Square Error):** Penalizes larger errors more heavily
- **R² (Coefficient of Determination):** Higher values indicate better fit (0-1 scale)

### Emissions Metrics

- **Emissions (kg CO₂e):** Carbon footprint of model training
- **Energy (kWh):** Total energy consumption
- **Duration (seconds):** Training time

### Trade-off Analysis

- **Efficient Models:** Low emissions + low error rates
- **Pareto Frontier:** Optimal balance between accuracy and sustainability
- **Country Variations:** Performance differences across European markets
