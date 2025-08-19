from __future__ import annotations
import random
import numpy as np
import pandas as pd
from pyDOE2 import lhs

def seed_population(dataset_path: str, bounds_lower, bounds_upper, size: int, cst_columns=None):
    if cst_columns is None:
        cst_columns = [f"CST Coeff {i}" for i in range(1, 9)]
    pop = []
    if dataset_path and dataset_path.strip():
        df = pd.read_excel(dataset_path)
        seed = df[cst_columns].dropna().values.tolist()
        random.shuffle(seed)
        pop.extend(seed[:size])
    remaining = size - len(pop)
    if remaining > 0:
        lb = np.asarray(bounds_lower, dtype=float)
        ub = np.asarray(bounds_upper, dtype=float)
        samples = lhs(lb.size, samples=remaining)  # [0,1]
        pts = lb + samples * (ub - lb)
        pop.extend(pts.tolist())
    return pop[:size]
