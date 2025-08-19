from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .geometry.cst import cst_airfoil

def plot_airfoil(cst, out: Path | None = None, show: bool = False):
    x = np.linspace(0.0, 1.0, 2001)
    _, yu, yl = cst_airfoil(np.asarray(cst, dtype=float), x, dz=0.0)
    plt.figure(figsize=(8,3))
    plt.plot(x, yu, label="Upper")
    plt.plot(x, yl, label="Lower")
    plt.axis("equal"); plt.xlim(0,1); plt.grid(True, linestyle=":"); plt.legend()
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def _handle_plot_show_save(fig, show: bool, save: bool, out_path: Path):
    """Saves and/or shows a Matplotlib figure based on config flags."""
    if save:
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    # Always close the figure to free up memory
    plt.close(fig)
