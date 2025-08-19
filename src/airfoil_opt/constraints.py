from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List
from .geometry.cst import AirfoilMetrics

@dataclass
class ConstraintLimits:
    max_thickness_limit: float   # optional upper bound on PEAK thickness
    min_thickness_limit: float   # LOWER bound on PEAK thickness (your 'min_t')
    min_t_pos: float
    max_t_pos: float
    min_le_radius: float
    max_le_radius: float
    max_camber: float
    min_camber: float
    max_c_pos: float
    min_c_pos: float

CONSTRAINT_NAMES = [
    "max_thick_upper_viol", "min_thick_lower_viol",
    "max_thick_pos_upper_viol", "min_thick_pos_lower_viol",
    "le_radius_upper_viol", "le_radius_lower_viol",
    "surface_crossover_viol", "max_camber_viol",
    "min_camber_viol", "max_c_pos_upper_viol", "min_c_pos_lower_viol"
]

def compute_constraint_violations(metrics: AirfoilMetrics, limits: ConstraintLimits) -> List[float]:
    v: list[float] = []
    # 1) peak thickness too large (optional cap)
    v.append(max(0.0, metrics.max_t - limits.max_thickness_limit))
    # 2) peak thickness too small (THIS enforces your min allowed max thickness)
    v.append(max(0.0, limits.min_thickness_limit - metrics.max_t))
    # 3) x-location of peak thickness (upper)
    v.append(max(0.0, metrics.pos_max_t - limits.max_t_pos))
    # 4) x-location of peak thickness (lower)
    v.append(max(0.0, limits.min_t_pos - metrics.pos_max_t))
    # 5) LE radius upper
    v.append(max(0.0, metrics.le_radius - limits.max_le_radius))
    # 6) LE radius lower
    v.append(max(0.0, limits.min_le_radius - metrics.le_radius))
    # 7) surface crossover: lower above upper anywhere
    diff = metrics.yl - metrics.yu
    v.append(float(np.max(np.clip(diff, 0.0, None))) if diff.size else 0.0)
    # 8) max camber
    v.append(max(0.0, metrics.max_camber - limits.max_camber))
    # 9) min camber
    v.append(max(0.0, limits.min_camber - metrics.max_camber))
    # 10) camber position upper
    v.append(max(0.0, metrics.pos_max_c - limits.max_c_pos))
    # 11) camber position lower
    v.append(max(0.0, limits.min_c_pos - metrics.pos_max_c))
    return v
