from __future__ import annotations
from deap import tools
from ..ga.selection import feasible_first

def check_bounds(min_b, max_b):
    """Decorator to enforce strict bounds after crossover/mutation."""
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in range(len(child)):
                    if child[i] > max_b[i]:
                        child[i] = max_b[i]
                    elif child[i] < min_b[i]:
                        child[i] = min_b[i]
            return offspring
        return wrapper
    return decorator

def register_operators(
    toolbox,
    bounds_lower,
    bounds_upper,
    cx_prob: float,
    mut_prob: float,
    eta_cx: float,
    eta_mut: float,
):
    # Bounded operators already clamp the genes to [low, up]
    toolbox.register(
        "mate",
        tools.cxSimulatedBinaryBounded,
        low=bounds_lower,
        up=bounds_upper,
        eta=eta_cx,
    )
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        low=bounds_lower,
        up=bounds_upper,
        eta=eta_mut,
        indpb=mut_prob,
    )
    toolbox.register("select", feasible_first)
    return toolbox
