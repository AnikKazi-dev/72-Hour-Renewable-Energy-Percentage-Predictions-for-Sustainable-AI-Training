from __future__ import annotations
from pathlib import Path
from contextlib import contextmanager
from codecarbon import EmissionsTracker
from .paths import EMISSIONS_DIR
from .countries import name_for as country_name_for
import os

@contextmanager
def emissions_tracker(task_name: str, output_file: str | None = None, measure_power_secs: int = 15):
    """Context manager returning (tracker, emissions_csv_path).

    The file path is computed up-front to avoid ambiguity when CodeCarbon
    appends to an existing CSV.
    """
    # Organize emissions as Results/emissions/<ModelName>/<season>/<run>.csv
    season = (os.environ.get("BIONET_SEASON") or "unknown").lower()
    model_name = (os.environ.get("ENTRY_MODEL_NAME") or os.environ.get("RUN_MODEL_NAME") or "model").replace(' ', '_')
    cc = (os.environ.get("COUNTRY_CODE") or os.environ.get("BIONET_COUNTRY_CODE") or "").upper()
    country = country_name_for(cc) if cc else "All"
    target_dir = EMISSIONS_DIR / model_name / country / season
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / (output_file or f"{task_name.replace(' ', '_')}_emissions.csv")
    tracker = EmissionsTracker(
        project_name=task_name,
    output_dir=str(target_dir),
    output_file=out.name,
        measure_power_secs=measure_power_secs,
        log_level="warning",
        save_to_file=True,
    )
    tracker.start()
    try:
        yield tracker, out
    finally:
        tracker.stop()
