from __future__ import annotations
import json, time, random, typer 
from pathlib import Path
import numpy as np
from deap import base, creator, tools
from ..config import Cfg
from ..paths import Paths
from ..models.surrogate import load_surrogate
from ..ga.seeding import seed_population
from ..ga.termination import HVStagnation, numeric_hv 
from ..ga.operators import register_operators
from ..ga.evaluation import evaluate_individual
from ..constraints import ConstraintLimits

from ..reporting import (
    pareto_scatter,
    hv_history,
    objectives_over_gens,
    diversity_over_gens,
    baseline_vs_optimized_ld,
    airfoil_evolution,
    optimized_vs_baseline_shape,
)

def _diversity(pop):
    import numpy as np
    if not pop:
        return 0.0
    arr = np.array([list(ind) for ind in pop], dtype=float)
    if arr.shape[0] < 2:
        return 0.0
    return float(np.mean(np.std(arr, axis=0)))

def _limits_from_baseline(cst_baseline, scale: float = 0.2):
    """
    Build constraint limits from the baseline airfoil, using a symmetric ±scale band.

    Semantics:
      - max_thickness_limit: upper bound on *peak* thickness
      - min_thickness_limit: lower bound on *peak* thickness  (your min_t)
      - max_t_pos/min_t_pos: x-location bounds for peak thickness
      - max_le_radius/min_le_radius: bounds on leading-edge radius
      - max_camber/min_camber: bounds on *maximum* camber (min_camber enforces "not too flat")
      - max_c_pos/min_c_pos: x-location bounds for max camber
    """
    import numpy as np
    from ..geometry.cst import evaluate_airfoil_geometry

    s = float(scale)
    m = evaluate_airfoil_geometry(np.asarray(cst_baseline, dtype=float))

    max_thickness_limit = m.max_t * (1 + s)
    min_thickness_limit = m.max_t * (1 - s)     # <-- lower bound on *peak* thickness (min_t)

    min_t_pos = m.pos_max_t * (1 - s)
    max_t_pos = m.pos_max_t * (1 + s)

    # LE radius: allow ±s band, but never below 0
    min_le_radius = max(0.0, m.le_radius * (1 - s))
    max_le_radius = m.le_radius * (1 + s)

    # Camber: bound the *maximum camber* symmetrically around the baseline
    max_camber = m.max_camber * (1 + s)
    min_camber = m.max_camber * (1 - s)         

    min_c_pos = m.pos_max_c * (1 - s)
    max_c_pos = m.pos_max_c * (1 + s)

    return ConstraintLimits(
        max_thickness_limit=max_thickness_limit,
        min_thickness_limit=min_thickness_limit,
        min_t_pos=min_t_pos,
        max_t_pos=max_t_pos,
        min_le_radius=min_le_radius,
        max_le_radius=max_le_radius,
        max_camber=max_camber,
        min_camber=min_camber,
        max_c_pos=max_c_pos,
        min_c_pos=min_c_pos,
    )

NUM_OBJ = 2
EPS = 0.0   # feasibility tolerance on violations

def feasible_ratio(pop):
    feas = [ind for ind in pop if all(v <= EPS for v in ind.fitness.values[NUM_OBJ:])]
    return len(feas) / max(1, len(pop))


def run_ga(cfg: Cfg, paths: Paths) -> Path:
    random.seed(cfg.seed.python); np.random.seed(cfg.seed.numpy)

    run_dir = paths.out_dir / time.strftime("%Y%m%d-%H%M%S")
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)

    model = load_surrogate(paths.model)

    n_genes = 8
    bounds_low = cfg.bounds.lower
    bounds_up  = cfg.bounds.upper
    try:
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0) + (-1.0,) * 11)
    except Exception:
        pass
    try:
        creator.create("Individual", list, fitness=creator.FitnessMulti)
    except Exception:
        pass
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initRepeat, creator.Individual, lambda: 0.0, n=n_genes)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    register_operators(toolbox, bounds_low, bounds_up, cfg.ga.crossover_prob, cfg.ga.base_indpb, cfg.ga.eta_cx, cfg.ga.eta_mut)

    pop = [creator.Individual(ind) for ind in seed_population(str(paths.dataset), bounds_low, bounds_up, cfg.ga.population)]
    hof = tools.ParetoFront()

    limits = _limits_from_baseline(cfg.airfoil.base_cst, scale=cfg.airfoil.scale)

    def _evaluate(ind):
        return evaluate_individual(ind, model, cfg.airfoil.aoa_list, cfg.airfoil.base_lds, cfg.airfoil.margin, limits)

    toolbox.register("evaluate", _evaluate)

    # Evaluate initial pop
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    pop = toolbox.select(pop, len(pop))  # NSGA-II sorting
    tools.emo.assignCrowdingDist(pop)

    # --- GEN 0 hypervolume on first K objectives (K=2 by default) ---
    K = 2   
    front0 = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
    hv_points0 = [tuple(-ind.fitness.values[i] for i in range(K)) for ind in front0]
    prev_hv = numeric_hv(hv_points0, tuple(cfg.termination.hv_reference_point))
    stagnation_count = 0

    log = {"meta": {"config": cfg.raw}, "history": []}

    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    with typer.progressbar(range(cfg.ga.max_generations), label="Optimizing Airfoils") as progress:
        for gen in progress:
            diversity = _diversity(pop)
            if diversity < cfg.ga.diversity_threshold:
                current_indpb = min(cfg.ga.max_indpb, cfg.ga.base_indpb * 2) 
            else:
                current_indpb = cfg.ga.base_indpb

            toolbox.register(
                "mutate",
                tools.mutPolynomialBounded,
                low=bounds_low,
                up=bounds_up,
                eta=cfg.ga.eta_mut,
                indpb=current_indpb, # Use the dynamically calculated probability
            )

            ratio = feasible_ratio(pop)
            if gen < 10 or ratio < 0.7:
                # early: mirror notebook (feasible-first parents)
                offspring = toolbox.select(pop, len(pop))
                offspring = [toolbox.clone(ind) for ind in offspring]
            else:
                pop = tools.selNSGA2(pop, len(pop))               # assign crowding_dist
                offspring = tools.selTournamentDCD(pop, len(pop))
                offspring = [toolbox.clone(ind) for ind in offspring]

            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if np.random.rand() < cfg.ga.crossover_prob:
                    toolbox.mate(ind1, ind2)
                    del ind1.fitness.values, ind2.fitness.values

            for mutant in offspring:
                toolbox.mutate(mutant)
                del mutant.fitness.values

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fits = list(map(toolbox.evaluate, invalid))
            for ind, fit in zip(invalid, fits):
                ind.fitness.values = fit

            combined_pop = pop + offspring
            pop[:] = toolbox.select(combined_pop, cfg.ga.population)
            tools.emo.assignCrowdingDist(pop)
            hof.update(pop)

            # Build Pareto front (maximization in DEAP via weights; keep only the first 2 objectives)
            front = [
                ind.fitness.values[:2]
                for ind in tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
            ]

            # ---- Hypervolume early stopping (DEAP expects MINIMIZATION) ----
            K = 2
            front = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
            hv_points = [tuple(-ind.fitness.values[i] for i in range(K)) for ind in front]
            current_hv = numeric_hv(hv_points, tuple(cfg.termination.hv_reference_point))

            if prev_hv > 0:
                rel_imp = (current_hv - prev_hv) / prev_hv
            else:
                rel_imp = 0.0

            if rel_imp >= cfg.termination.stagnation_tolerance:
                stagnation_count = 0
            else:
                stagnation_count += 1

            prev_hv = current_hv
            
            if stagnation_count >= cfg.termination.stagnation_generations:
                typer.echo(f"\n Converged at generation {gen} due to hypervolume stagnation.")
                break

            # Per-gen stats
            f1_vals = [ind.fitness.values[0] for ind in pop]
            f2_vals = [ind.fitness.values[1] for ind in pop]
            f1_best = float(np.max(f1_vals)) if f1_vals else -1
            f2_best = float(np.max(f2_vals)) if f2_vals else -1
            best_by_f1_idx = int(np.argmax(f1_vals)) if f1_vals else 0
            best_by_f1 = list(pop[best_by_f1_idx]) if pop else []

            progress.label = f"Gen {gen + 1:3d} | Best F1: {f1_best:6.2f} | Best F2: {f2_best:6.2f}"

            log["history"].append({
                "gen": gen,
                "front": front,                      # list of [f1,f2] on the first front
                "hv_current": current_hv,       # hv in minimization space
                "diversity": _diversity(pop),
                "f1_best": float(np.max(f1_vals)) if f1_vals else None,
                "f1_mean": float(np.mean(f1_vals)) if f1_vals else None,
                "f2_best": float(np.max(f2_vals)) if f2_vals else None,
                "f2_mean": float(np.mean(f2_vals)) if f2_vals else None,
                "best_individual": best_by_f1,       # CST coeffs of best-by-f1 this gen
            })

            if stagnation_count >= cfg.termination.stagnation_generations:
                break

    (run_dir / "meta.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    import pandas as pd
    pd.DataFrame([
        {"f1": ind.fitness.values[0], "f2": ind.fitness.values[1], **{f"c{i}": v for i, v in enumerate(ind, 1)}}
        for ind in hof
    ]).to_csv(run_dir / "pareto.csv", index=False)

    # --- REPORTS / PLOTS ---
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # select a "best" individual (prefer HOF, else last gen best)
    if len(hof) > 0:
        best = max(hof, key=lambda ind: ind.fitness.values[0])
        best_coeffs = list(best)
    else:
        best_coeffs = log["history"][-1]["best_individual"] if log["history"] else None

    # make the plots
    pareto_scatter(hof, plots_dir, show=cfg.plotting.show, save=cfg.plotting.save)
    hv_history(log["history"], plots_dir, show=cfg.plotting.show, save=cfg.plotting.save)
    objectives_over_gens(log["history"], plots_dir, show=cfg.plotting.show, save=cfg.plotting.save)
    diversity_over_gens(log["history"], plots_dir, show=cfg.plotting.show, save=cfg.plotting.save)
    if best_coeffs is not None:
        baseline_vs_optimized_ld(cfg, model, best_coeffs, plots_dir, show=cfg.plotting.show, save=cfg.plotting.save)
        optimized_vs_baseline_shape(cfg.airfoil.base_cst, best_coeffs, plots_dir, show=cfg.plotting.show, save=cfg.plotting.save)
    airfoil_evolution(log["history"], cfg.airfoil.base_cst, plots_dir, show=cfg.plotting.show, save=cfg.plotting.save)

    return run_dir
