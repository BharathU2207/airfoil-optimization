from __future__ import annotations
import numpy as np
import joblib
from pathlib import Path
from typing import Protocol

class Surrogate(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray: ...

def load_surrogate(model_path: Path):
    model = joblib.load(model_path)
    if not hasattr(model, "predict"):
        raise TypeError("Loaded object does not implement .predict")
    return model

def predict_ld(model, cst: np.ndarray, aoa: float) -> float:
    feats = np.concatenate([[aoa], np.asarray(cst, dtype=float)]).reshape(1, -1)
    pred = float(model.predict(feats)[0])
    return pred if np.isfinite(pred) else 0.0
