from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from scipy.special import comb

@dataclass
class AirfoilMetrics:
    yu: np.ndarray
    yl: np.ndarray
    max_t: float       # peak thickness
    pos_max_t: float   # x-location of peak thickness
    min_t: float       # minimum thickness anywhere (not used in the min-peak constraint)
    max_camber: float
    pos_max_c: float
    le_radius: float

def class_shape_function(w: np.ndarray, x: np.ndarray, N1: float = 0.5, N2: float = 1.0, dz: float = 0.0) -> np.ndarray:
    C = np.power(x, N1) * np.power(1 - x, N2)
    n = len(w) - 1
    S = np.zeros_like(x)
    for j in range(n + 1):
        S += w[j] * comb(n, j) * np.power(x, j) * np.power(1 - x, n - j)
    return (C * S) + x * dz

def cst_airfoil(w: np.ndarray, x: np.ndarray, dz: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    wl = np.asarray(w[: len(w)//2], dtype=float)
    wu = np.asarray(w[len(w)//2 :], dtype=float)
    yl = class_shape_function(wl, x, N1=0.5, N2=1.0, dz=-dz)
    yu = class_shape_function(wu, x, N1=0.5, N2=1.0, dz=dz)
    # force TE closure
    yu[-1] = 0.0; yl[-1] = 0.0
    return np.concatenate([yu, yl]), yu, yl

def compute_camber(yu: np.ndarray, yl: np.ndarray) -> np.ndarray:
    return (yu + yl) * 0.5

def compute_thickness(yu: np.ndarray, yl: np.ndarray) -> np.ndarray:
    return yu - yl

def estimate_le_radius(x: np.ndarray, yu: np.ndarray, num_points: int = 10) -> float:
    from scipy.optimize import curve_fit
    def circle(x, r, x0, y0):
        return np.sqrt(np.maximum(r**2 - (x - x0)**2, 0.0)) + y0
    x_upper = x[:num_points]
    y_upper = yu[:num_points]
    try:
        popt, _ = curve_fit(circle, x_upper, y_upper, maxfev=2000)
        return float(abs(popt[0]))
    except Exception:
        return 0.0

def evaluate_airfoil_geometry(cst_coeffs: np.ndarray, dz: float = 0.0, x: np.ndarray | None = None) -> AirfoilMetrics:
    if x is None:
        x = np.linspace(0.0, 1.0, 2001)
    _, yu, yl = cst_airfoil(cst_coeffs, x, dz=dz)
    thickness = compute_thickness(yu, yl)
    camber = compute_camber(yu, yl)
    max_t_idx = int(np.argmax(thickness))
    max_c_idx = int(np.argmax(camber))
    max_t = float(thickness[max_t_idx])
    pos_max_t = float(x[max_t_idx])
    min_t = float(np.min(thickness))
    max_camber = float(camber[max_c_idx])
    pos_max_c = float(x[max_c_idx])
    le_radius = estimate_le_radius(x[:51], yu[:51])
    return AirfoilMetrics(yu=yu, yl=yl, max_t=max_t, pos_max_t=pos_max_t, min_t=min_t,
                          max_camber=max_camber, pos_max_c=pos_max_c, le_radius=le_radius)
