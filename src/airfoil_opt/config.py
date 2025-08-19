from __future__ import annotations
import yaml
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class AirfoilCfg:
    base_cst: list[float]
    base_lds: list[float]
    aoa_list: list[int]
    margin: float
    scale: float

@dataclass
class BoundsCfg:
    lower: list[float]
    upper: list[float]

@dataclass
class GACfg:
    population: int
    max_generations: int
    seed_dataset_fraction: float
    crossover_prob: float
    base_indpb: float
    max_indpb: float 
    diversity_threshold: float 
    eta_cx: float
    eta_mut: float

@dataclass
class TerminationCfg:
    hv_reference_point: list[float]
    stagnation_tolerance: float
    stagnation_generations: int

@dataclass
class SeedCfg:
    python: int
    numpy: int
    deap: int

@dataclass
class PlotCfg:
    show: bool
    save: bool

@dataclass
class Cfg:
    raw: Dict[str, Any]
    airfoil: AirfoilCfg
    bounds: BoundsCfg
    ga: GACfg
    seed: SeedCfg
    plotting: PlotCfg
    termination: TerminationCfg

def load_config(path: str) -> Cfg:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Cfg(
        raw=raw,
        airfoil=AirfoilCfg(**raw["airfoil"]),
        bounds=BoundsCfg(**raw["bounds"]),
        ga=GACfg(**raw["ga"]),
        seed=SeedCfg(**raw["seed"]),
        plotting=PlotCfg(**raw["plotting"]),
        termination=TerminationCfg(**raw["termination"]),
    )
