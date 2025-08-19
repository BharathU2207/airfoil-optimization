from __future__ import annotations
from typing import List, Tuple
import numpy as np
try:
    from deap.tools._hypervolume import hv as hv_module
    numeric_hv = hv_module.hypervolume
except Exception:
    from deap.tools._hypervolume.pyhv import hypervolume as numeric_hv

class HVStagnation:
    def __init__(self, reference_point, tolerance: float, patience: int):
        self.ref = tuple(reference_point)
        self.tol = float(tolerance)
        self.patience = int(patience)
        self.best = -np.inf
        self.bad = 0

    def step(self, front: List[Tuple[float, ...]]) -> bool:
        if not front:
            return False
        hv = numeric_hv(front, self.ref)
        if hv > self.best + self.tol:
            self.best = hv
            self.bad = 0
        else:
            self.bad += 1
        return self.bad >= self.patience
