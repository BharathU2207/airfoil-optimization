from __future__ import annotations
from deap import tools
from ..ga.selection import feasible_first

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
