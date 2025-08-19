# src/airfoil_opt/reporting.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Sequence, Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt

from .plotting import plot_airfoil, _handle_plot_show_save
from .geometry.cst import cst_airfoil
from .models.surrogate import predict_ld


# ---------- small helpers ----------

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _diversity(pop) -> float:
    """Mean per-gene std across the population."""
    if not pop:
        return 0.0
    arr = np.array([list(ind) for ind in pop], dtype=float)
    if arr.shape[0] < 2:
        return 0.0
    return float(np.mean(np.std(arr, axis=0)))


# ---------- figure writers ----------

def pareto_scatter(hof, out_dir: Path, show, save) -> None:
    out_dir = _ensure_dir(out_dir)
    if len(hof) == 0:
        return
    f1 = [ind.fitness.values[0] for ind in hof]
    f2 = [ind.fitness.values[1] for ind in hof]
    fig = plt.figure(figsize=(6, 5))
    plt.scatter(f1, f2, s=18)
    plt.xlabel("Peak L/D (f1)")
    plt.ylabel("Cumulative Performance Gain (f2)")
    plt.title("Pareto front")
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    _handle_plot_show_save(
        fig, 
        show=show, 
        save=save, 
        out_path=out_dir / "pareto_front.png"
    )


def hv_history(history: List[Dict[str, Any]], out_dir: Path, show, save) -> None:
    out_dir = _ensure_dir(out_dir)
    if not history:
        return
    hv_vals = [h.get("hv_current") for h in history]
    if not hv_vals:
        return
    fig = plt.figure(figsize=(6, 4))
    plt.plot(range(len(hv_vals)), hv_vals, marker="o", linewidth=1)
    plt.xlabel("Generation")
    plt.ylabel("Best HV (minimization space)")
    plt.title("Hypervolume progress")
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    _handle_plot_show_save(fig, show, save, out_dir / "hv_history.png")


def objectives_over_gens(history: List[Dict[str, Any]], out_dir: Path, show, save) -> None:
    out_dir = _ensure_dir(out_dir)
    if not history:
        return
    gens = [h["gen"] for h in history]
    f1_best = [h.get("f1_best") for h in history]
    f1_mean = [h.get("f1_mean") for h in history]
    f2_best = [h.get("f2_best") for h in history]
    f2_mean = [h.get("f2_mean") for h in history]

    # f1
    fig = plt.figure(figsize=(7, 4))
    plt.plot(gens, f1_best, label="Peak L/D best")
    plt.plot(gens, f1_mean, label="Peak L/D mean", linestyle="--")
    plt.xlabel("Generation")
    plt.ylabel("Peak L/D")
    plt.title("Peak L/D over generations")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    _handle_plot_show_save(fig, show, save, out_dir / "peak_ld_over_gens.png")

    # f2
    fig = plt.figure(figsize=(7, 4))
    plt.plot(gens, f2_best, label="CPG best")
    plt.plot(gens, f2_mean, label="CPG mean", linestyle="--")
    plt.xlabel("Generation")
    plt.ylabel("Cumulative Performance Gain")
    plt.title("Cumulative Performance Gain over generations")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    _handle_plot_show_save(fig, show, save, out_dir / "CPG_over_gens.png")

def diversity_over_gens(history: List[Dict[str, Any]], out_dir: Path, show, save) -> None:
    out_dir = _ensure_dir(out_dir)
    if not history:
        return
    gens = [h["gen"] for h in history]
    vals = [h.get("diversity", 0.0) for h in history]
    fig = plt.figure(figsize=(7, 4))
    plt.plot(gens, vals, marker=".")
    plt.xlabel("Generation")
    plt.ylabel("Mean per-gene std")
    plt.title("Diversity over generations")
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    _handle_plot_show_save(fig, show, save, out_dir / "diversity_over_gens.png")

def baseline_vs_optimized_ld(cfg, model, best_coeffs: Sequence[float], out_dir: Path, show, save) -> None:
    """Plot baseline L/D vs optimized L/D (no margin)."""
    out_dir = _ensure_dir(out_dir)
    aoa = np.asarray(cfg.airfoil.aoa_list, dtype=float)
    baseline = np.asarray(cfg.airfoil.base_lds, dtype=float)
    preds = np.array([predict_ld(model, best_coeffs, a) for a in aoa], dtype=float)

    fig = plt.figure(figsize=(7, 4))
    plt.plot(aoa, baseline, label="Baseline L/D")
    plt.plot(aoa, preds, label="Optimized L/D")
    plt.xlabel("AoA")
    plt.ylabel("L/D")
    plt.title("Baseline vs Optimized L/D (no margin)")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    _handle_plot_show_save(fig, show, save, out_dir / "baseline_vs_optimized_ld.png")

def airfoil_evolution(history: List[Dict[str, Any]], base_cst: Sequence[float], out_dir: Path, show, save) -> None:
    """Plot airfoil shape evolution for gens {0,5,10,15} plus 6 evenly spaced to last."""
    out_dir = _ensure_dir(out_dir)
    if not history:
        return

    def _select_gens(hist):
        last = hist[-1]["gen"]
        seeds = {0, 5, 10, 15}
        if last > 15:
            pts = np.linspace(15, last, 7)[1:]  # 6 evenly spaced
            seeds.update(np.round(pts).astype(int))
        return sorted(g for g in seeds if 0 <= g <= last)

    airfoils_over_gens = {h["gen"]: h.get("best_individual") for h in history if h.get("best_individual")}
    gens_to_plot = _select_gens(history)
    if not gens_to_plot:
        return

    x = np.linspace(0.0, 1.0, 2001)
    _, yu_og, yl_og = cst_airfoil(np.asarray(base_cst, dtype=float), x, dz=0.0)

    rows, cols = 2, 5
    fig, axes = plt.subplots(rows, cols, figsize=(20, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    for i in range(rows * cols):
        axes[i].axis("off")

    for i, gen in enumerate(gens_to_plot[: rows * cols]):
        coeffs = airfoils_over_gens.get(gen)
        if coeffs is None:
            continue
        _, yu, yl = cst_airfoil(np.asarray(coeffs, dtype=float), x, dz=0.0)
        ax = axes[i]
        ax.axis("on")
        ax.plot(x, yu_og, linestyle="--", linewidth=1, label="Baseline" if i == 0 else None)
        ax.plot(x, yl_og, linestyle="--", linewidth=1)
        ax.plot(x, yu, label=f"Gen {gen}")
        ax.plot(x, yl)
        ax.set_title(f"Generation {gen}")
        ax.grid(True, linestyle=":")
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle("Evolution of Airfoil Shape Over Generations")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _handle_plot_show_save(fig, show, save, out_dir / "airfoil_evolution.png")


def optimized_vs_baseline_shape(base_cst: Sequence[float], best_coeffs: Sequence[float], out_dir: Path, show, save) -> None:
    """Plot final optimized airfoil vs baseline (shape)."""
    out_dir = _ensure_dir(out_dir)
    x = np.linspace(0.0, 1.0, 2001)
    _, yu_og, yl_og = cst_airfoil(np.asarray(base_cst, dtype=float), x, dz=0.0)
    _, yu_t, yl_t = cst_airfoil(np.asarray(best_coeffs, dtype=float), x, dz=0.0)

    fig = plt.figure(figsize=(8, 3))
    plt.plot(x, yu_og, linestyle="--", label="Baseline")
    plt.plot(x, yl_og, linestyle="--")
    plt.plot(x, yu_t, label="Optimized")
    plt.plot(x, yl_t)
    plt.axis("equal")
    plt.xlim(0, 1)
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    _handle_plot_show_save(fig, show, save, out_dir / "optimized_vs_baseline_shape.png")
