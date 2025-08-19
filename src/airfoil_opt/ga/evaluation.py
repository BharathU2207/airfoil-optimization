from __future__ import annotations
import numpy as np
from typing import Sequence
from ..geometry.cst import evaluate_airfoil_geometry
from ..constraints import compute_constraint_violations, ConstraintLimits
from ..objectives import ld_peak, cumulative_performance_gain
from ..models.surrogate import Surrogate

def evaluate_individual(ind: Sequence[float], model: Surrogate, aoa_list, base_curve, margin: float, limits: ConstraintLimits):
    cst = np.asarray(ind, dtype=float)
    metrics = evaluate_airfoil_geometry(cst)
    f1 = ld_peak(model, cst, aoa_list)
    f2 = cumulative_performance_gain(model, cst, aoa_list, np.asarray(base_curve, dtype=float), margin)
    viol = compute_constraint_violations(metrics, limits)
    return (f1, f2, *viol)
