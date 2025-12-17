from __future__ import annotations
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import asdict
from deap import base, creator, tools
import typer

from ..config import Cfg
from ..paths import Paths
from ..models.surrogate import load_surrogate, predict_ld
from ..ga.seeding import seed_population
from ..ga.operators import register_operators, check_bounds
from ..geometry.cst import evaluate_airfoil_geometry, cst_airfoil
from ..constraints import compute_constraint_violations, ConstraintLimits
import json
import matplotlib.pyplot as plt

def save_airfoil_dat(path: Path, x: np.ndarray, yu: np.ndarray, yl: np.ndarray, name: str):
    """Saves the airfoil in standard Selig .dat format for XFOIL/CFD."""
    with open(path, "w") as f:
        f.write(f"{name}\n")

        for i in range(len(x)):
            f.write(f" {x[i]:.6f}  {yu[i]:.6f}\n")
        for i in range(len(x)-2, -1, -1):
             f.write(f" {x[i]:.6f}  {yl[i]:.6f}\n")

def _diversity(pop):
    if not pop: return 0.0
    arr = np.array([list(ind) for ind in pop], dtype=float)
    if arr.shape[0] < 2: return 0.0
    return float(np.mean(np.std(arr, axis=0)))

def run_inverse_optimization(cfg: Cfg, paths: Paths) -> Path:
    # Setup
    random.seed(cfg.seed.python)
    np.random.seed(cfg.seed.numpy)
    
    out_dir = paths.out_dir / f"inverse_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model = load_surrogate(paths.model)
    
    # Config Extraction
    TARGET_AOA = cfg.inverse.target_aoa
    TARGET_LD = cfg.inverse.target_ld
    POP_SIZE = cfg.ga.population
    MAX_GEN = cfg.ga.max_generations
    
    # Create Limits object from Config
    limits = ConstraintLimits(**asdict(cfg.inverse.limits))

    print(f"GLOBAL INVERSE DESIGN")
    print(f"Target: L/D={TARGET_LD} @ {TARGET_AOA} deg")
    
    # GA Setup
    # Fitness: Minimize Error (1 obj) + Minimize Constraints (11 objs)
    weights = (-1.0,) + (-1.0,) * 11
    
    if hasattr(creator, "InverseFitness"): del creator.InverseFitness
    if hasattr(creator, "InverseInd"): del creator.InverseInd
    creator.create("InverseFitness", base.Fitness, weights=weights)
    creator.create("InverseInd", list, fitness=creator.InverseFitness)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initRepeat, creator.InverseInd, lambda: 0.0, n=8)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    register_operators(toolbox, cfg.bounds.lower, cfg.bounds.upper, 
                       cfg.ga.crossover_prob, cfg.ga.base_indpb, 
                       cfg.ga.eta_cx, cfg.ga.eta_mut)

    # Apply Bounds Check
    toolbox.decorate("mate", check_bounds(cfg.bounds.lower, cfg.bounds.upper))
    toolbox.decorate("mutate", check_bounds(cfg.bounds.lower, cfg.bounds.upper))
    toolbox.register("mate", tools.cxBlend, alpha=0.5)

    def evaluate(ind):
        cst = np.asarray(ind, dtype=float)
        pred_ld = predict_ld(model, cst, TARGET_AOA)
        error = abs(pred_ld - TARGET_LD)
        metrics = evaluate_airfoil_geometry(cst)
        viol = compute_constraint_violations(metrics, limits)
        return (error, *viol)

    toolbox.register("evaluate", evaluate)
    toolbox.register("select", tools.selNSGA2)

    # Initialization
    pop = [creator.InverseInd(ind) for ind in seed_population(str(paths.dataset), cfg.bounds.lower, cfg.bounds.upper, POP_SIZE)]
    
    # Evaluate Initial
    fits = toolbox.map(toolbox.evaluate, pop)
    for fit, ind in zip(fits, pop):
        ind.fitness.values = fit
    tools.emo.assignCrowdingDist(pop)
    
    history_error = []

    # Main Loop
    with typer.progressbar(range(MAX_GEN), label="Inverse Design") as progress:
        for gen in progress:
            diversity = _diversity(pop)
            if diversity < cfg.ga.diversity_threshold:
                current_indpb = min(cfg.ga.max_indpb, cfg.ga.base_indpb * 2) 
            else:
                current_indpb = cfg.ga.base_indpb

            toolbox.register("mutate", tools.mutPolynomialBounded, low=cfg.bounds.lower, up=cfg.bounds.upper, eta=cfg.ga.eta_mut, indpb=current_indpb)
            toolbox.decorate("mutate", check_bounds(cfg.bounds.lower, cfg.bounds.upper))

            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cfg.ga.crossover_prob:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values, child2.fitness.values

            for mutant in offspring:
                toolbox.mutate(mutant)
                del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fits = toolbox.map(toolbox.evaluate, invalid_ind)
            for fit, ind in zip(fits, invalid_ind):
                ind.fitness.values = fit

            combined_pop = pop + offspring
            pop[:] = toolbox.select(combined_pop, POP_SIZE)
            tools.emo.assignCrowdingDist(pop)
            
            # Stats
            feasible_inds = [ind for ind in pop if all(v <= 1e-6 for v in ind.fitness.values[1:])]
            if feasible_inds:
                best_feas = min(feasible_inds, key=lambda ind: ind.fitness.values[0])
                best_err = best_feas.fitness.values[0]
                status = "FEASIBLE"
            else:
                best_raw = min(pop, key=lambda ind: ind.fitness.values[0])
                best_err = best_raw.fitness.values[0]
                status = "INFEASIBLE"
                
            history_error.append({"Generation": gen, "Error": best_err, "Status": status})
            progress.label = f"Gen {gen}: Best Err {best_err:.4f} [{status}]"

    # Save Results
    if feasible_inds:
        best_ind = min(feasible_inds, key=lambda ind: ind.fitness.values[0])
    else:
        best_ind = min(pop, key=lambda ind: ind.fitness.values[0])
    
    best_cst = np.asarray(best_ind, dtype=float)
    final_error = best_ind.fitness.values[0]

    meta_data = {
        "config": cfg.raw,
        "results": {
            "final_error": final_error,
            "best_cst": list(best_cst),
            "status": "FEASIBLE" if feasible_inds else "INFEASIBLE"
        }
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta_data, f, indent=2)

    # CST coefficients 
    cst_data = {f"c{i+1}": [val] for i, val in enumerate(best_cst)}
    cst_data["error"] = [final_error]
    pd.DataFrame(cst_data).to_csv(out_dir / "optimized_cst.csv", index=False)

    # Convergence 
    df_conv = pd.DataFrame(history_error)
    df_conv.to_csv(out_dir / "convergence.csv", index=False)
    
    plt.figure(figsize=(8, 6))
    plt.plot(df_conv["Generation"], df_conv["Error"], marker='o', markersize=3)
    # plt.yscale('log') 
    plt.xlabel("Generation")
    plt.ylabel("L/D Error")
    plt.title(f"Convergence History (Target L/D={TARGET_LD})")
    plt.grid(True, which="both", linestyle=":")
    plt.savefig(out_dir / "convergence_plot.png")
    plt.close()
    
    # Save Geometry
    x_coords = np.linspace(0.0, 1.0, 201)
    _, yu, yl = cst_airfoil(best_cst, x_coords, dz=0.0)
    
    # CSV
    pd.DataFrame({"x": x_coords, "yu": yu, "yl": yl}).to_csv(out_dir / "geometry.csv", index=False)
    
    # DAT
    save_airfoil_dat(out_dir / "optimized.dat", x_coords, yu, yl, f"InvDesign_LD{TARGET_LD}")

    # Image
    plt.figure(figsize=(8, 3))
    plt.plot(x_coords, yu, 'k-', linewidth=1.5)
    plt.plot(x_coords, yl, 'k-', linewidth=1.5)
    plt.axis('equal')
    plt.grid(True, linestyle=":")
    plt.title(f"Optimized Airfoil (Error: {final_error:.4f})")
    plt.savefig(out_dir / "geometry_plot.png")
    plt.close()

    # Validation polar 
    val_aoas = np.linspace(-4, 8, 13) 
    val_lds = [predict_ld(model, best_cst, a) for a in val_aoas]
    
    pd.DataFrame({"AoA": val_aoas, "Predicted_LD": val_lds}).to_csv(out_dir / "validation_polar.csv", index=False)
    
    plt.figure(figsize=(6, 4))
    plt.plot(val_aoas, val_lds, 'b.-', label="Inverse Design")
    plt.plot(TARGET_AOA, TARGET_LD, 'rx', markersize=10, markeredgewidth=2, label="Target")
    plt.xlabel("Angle of Attack (deg)")
    plt.ylabel("L/D Ratio")
    plt.title("Validation Polar")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir / "validation_polar.png")
    plt.close()

    typer.echo(f"Inverse Design Complete. Results saved to {out_dir}")
    return out_dir