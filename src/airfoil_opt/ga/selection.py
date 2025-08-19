# src/airfoil_opt/ga/selection.py
from __future__ import annotations
from deap import tools

def feasible_first(pop, k, num_obj: int = 2, epsilon: float = 0.0):
    """Feasible-first selection.
    - Take all feasibles (violations <= epsilon), rank them by NSGA-II on objectives only.
    - If not enough, fill the rest with infeasibles sorted by total violation (asc).
    """
    feas = [ind for ind in pop if all(v <= epsilon for v in ind.fitness.values[num_obj:])]
    infeas = [ind for ind in pop if ind not in feas]

    if len(feas) >= k:
        return tools.selNSGA2(feas, k)  # rank by (f1,f2), preserve diversity
    feas_ranked = tools.selNSGA2(feas, len(feas)) if feas else []
    need = k - len(feas_ranked)
    infeas_sorted = sorted(infeas, key=lambda ind: sum(ind.fitness.values[num_obj:]))
    return feas_ranked + infeas_sorted[:need]
