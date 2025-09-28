import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
MODELS_DIR = ROOT / "Models"
RESULTS_DIR = ROOT / "Results"
IMAGES_DIR = RESULTS_DIR / "images"
METRICS_DIR = RESULTS_DIR / "metrics"
REPORTS_DIR = RESULTS_DIR / "reports"
EMISSIONS_DIR = RESULTS_DIR / "emissions"

for d in [DATA_DIR, MODELS_DIR, IMAGES_DIR, METRICS_DIR, REPORTS_DIR, EMISSIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
