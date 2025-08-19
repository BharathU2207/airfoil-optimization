from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class Paths:
    repo_root: Path
    out_dir: Path
    dataset: Path
    model: Path

    @staticmethod
    def from_config(repo_root: Path, cfg: dict) -> "Paths":
        p = cfg["paths"]
        out_dir = (repo_root / p["out_dir"]).resolve()
        return Paths(
            repo_root=repo_root,
            out_dir=out_dir,
            dataset=(repo_root / p["dataset"]).resolve(),
            model=(repo_root / p["model"]).resolve(),
        )
