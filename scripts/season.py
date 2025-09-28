from __future__ import annotations
import os
from pathlib import Path


def resolve_season(default: str | None = None) -> str:
    s = os.environ.get("BIONET_SEASON") or (default or "winter")
    s = str(s).strip().lower()
    return "summer" if s.startswith("sum") else "winter"


def months_for(season: str) -> list[int]:
    s = str(season).strip().lower()
    if s == "summer":
        return [4, 5, 6, 7, 8, 9]
    return [10, 11, 12, 1, 2, 3]


def data_path(name: str) -> Path:
    from .paths import DATA_DIR
    return DATA_DIR / name
