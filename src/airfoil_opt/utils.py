from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np

def set_seeds(py: int, np_seed: int):
    random.seed(py); np.random.seed(np_seed)

def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
