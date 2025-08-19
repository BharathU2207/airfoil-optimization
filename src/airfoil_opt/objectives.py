from __future__ import annotations
import numpy as np
from typing import Iterable
from .models.surrogate import predict_ld

def ld_peak(model, cst: np.ndarray, aoa_list: Iterable[float]) -> float:
    vals = [predict_ld(model, cst, a) for a in aoa_list]
    return float(np.max(vals))

def cumulative_performance_gain(model, cst: np.ndarray, aoa_list: Iterable[float],
                                base_curve: np.ndarray, margin: float) -> float:
    preds = np.array([predict_ld(model, cst, a) for a in aoa_list], dtype=float)
    basep = np.asarray(base_curve, dtype=float) + float(margin)
    gains = np.clip(preds - basep, a_min=0.0, a_max=None)
    return float(np.sum(gains))
