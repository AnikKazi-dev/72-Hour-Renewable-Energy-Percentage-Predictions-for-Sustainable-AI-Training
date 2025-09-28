from __future__ import annotations
from typing import Dict, Iterable
import numpy as np


def _to_np(a: Iterable[float]) -> np.ndarray:
    return np.asarray(list(a), dtype=float)


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    yt, yp = _to_np(y_true), _to_np(y_pred)
    return float(np.mean(np.abs(yt - yp)))


def mse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    yt, yp = _to_np(y_true), _to_np(y_pred)
    return float(np.mean((yt - yp) ** 2))


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true: Iterable[float], y_pred: Iterable[float], eps: float = 1e-8) -> float:
    yt, yp = _to_np(y_true), _to_np(y_pred)
    denom = np.maximum(np.abs(yt), eps)
    return float(np.mean(np.abs((yt - yp) / denom)))


def as_dict(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }
