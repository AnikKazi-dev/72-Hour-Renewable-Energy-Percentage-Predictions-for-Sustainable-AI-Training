from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict
import os
from .paths import IMAGES_DIR, METRICS_DIR, REPORTS_DIR
from .countries import name_for as country_name_for


@dataclass
class Metrics:
    mae: float | None = None
    mse: float | None = None
    rmse: float | None = None
    mape: float | None = None


class ResultsManager:
    def __init__(self, run_name: str, model_name: str | None = None, season: str | None = None):
        safe = run_name.replace(" ", "_")
        self.run_name = safe
        self.model_name = (model_name or "model").replace(" ", "_")
        self.season = (season or "unknown").lower()

        # Country folder (human-readable) based on env COUNTRY_CODE
        cc = (os.environ.get("COUNTRY_CODE") or os.environ.get("BIONET_COUNTRY_CODE") or "").upper()
        self.country = country_name_for(cc) if cc else "All"

        # Directories organized by ModelName/Country/Season
        self.image_dir = IMAGES_DIR / self.model_name / self.country / self.season
        self.metrics_dir = METRICS_DIR / self.model_name / self.country / self.season
        self.report_dir = REPORTS_DIR / self.model_name / self.country / self.season
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.report_path = self.report_dir / f"{safe}.json"

    def save_metrics(self, metrics: Dict[str, Any] | Metrics, fname: str = "metrics.json") -> Path:
        payload = asdict(metrics) if isinstance(metrics, Metrics) else dict(metrics)
        out = self.metrics_dir / fname
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return out

    def append_report(self, payload: Dict[str, Any]) -> Path:
        report: Dict[str, Any] = {}
        if self.report_path.exists():
            try:
                report = json.loads(self.report_path.read_text(encoding="utf-8"))
            except Exception:
                report = {}
        report.update(payload)
        with self.report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return self.report_path

    def image_path(self, fname: str) -> Path:
        return self.image_dir / fname
