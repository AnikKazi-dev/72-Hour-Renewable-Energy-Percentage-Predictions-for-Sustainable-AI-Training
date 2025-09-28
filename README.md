# BioNet Python Project

This project was migrated from multiple Jupyter notebooks into a clean Python project layout with utilities to convert notebooks, store results, and track carbon emissions.

## Structure
- Data/: input data (csv, parquet, etc.)
- Data_layer/: loaders and preprocessing utilities
- Models/: .py versions of each former notebook model
- Results/
  - images/: saved plots
  - metrics/: JSON/CSV metrics (MAE, MSE, etc.)
  - reports/: aggregated reports
  - emissions/: CodeCarbon emission logs
- scripts/
  - convert_notebooks.py: convert all .ipynb into .py with nbconvert
  - run_with_emissions.py: run any training/inference entrypoint under CodeCarbon tracking
  - demo_run.py: minimal example that saves an image, metrics, and a report
  - metrics.py: quick metric helpers (MAE, MSE, RMSE, MAPE)

## Quick start
1) Install requirements (inside your env)

```powershell
pip install -r requirements.txt
```

2) Convert all notebooks to .py under `Models/`:

```powershell
python -m scripts.convert_notebooks
```

3) Run any script with emissions tracking. Example demo:

```powershell
python -m scripts.run_with_emissions scripts/demo_run.py --run-name demo
```

To run a converted model (e.g., `Models/Transformer_Model.py`):

```powershell
python -m scripts.run_with_emissions Models/Transformer_Model.py --run-name transformer --season winter
```

Images from matplotlib are auto-redirected to `Results/images/<run-name>/`.

## Seasonal orchestrators

- Run all models on the winter dataset (with CodeCarbon):

```powershell
python .\main_winter.py --quick --filter Transformer
```

- Run all models on the summer dataset:

```powershell
python .\main_summer.py --quick
```

Flags:
- `--filter <substr>` to limit which `Models/*.py` scripts run.
- `--quick` to cap Keras epochs for fast smoke tests.

## Run all models automatically

Use the new orchestrator `main.py` to run every `Models/*.py` with CodeCarbon tracking and per-run reports:

```powershell
python main.py
```

Filter to a subset:

```powershell
python main.py --filter Transformer
```

## Notes
- CodeCarbon will create `*.csv` files in `Results/emissions` by default (see `codecarbon.config`).
- Use `scripts/results_manager.py` in your training/inference scripts to store images/metrics and append to a run report.
