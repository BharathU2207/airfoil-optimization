from __future__ import annotations
import typer
from pathlib import Path
from .config import load_config
from .paths import Paths
from .ga.run import run_ga

app = typer.Typer(add_completion=False, help="Airfoil optimisation CLI (CST + surrogate + GA).")

@app.command()
def optimize(config: str = typer.Option("configs/default.yaml", help="Path to YAML config."),
            population: int = typer.Option(None, "--population", "-p", help="Override the population size from the config file."),
            generations: int = typer.Option(None, "--generation", "-g", help="Override the max number of generations from the config file.")
            ):
    cfg = load_config(config)
    if population is not None:
        cfg.ga.population = population
        cfg.raw['ga']['population'] = population
        typer.echo(f"CLI Override: Population size set to {population}")
    if generations is not None:
        cfg.ga.max_generations = generations
        cfg.raw['ga']['max_generations'] = generations
        typer.echo(f"CLI Override: Max generations set to {generations}")

    repo_root = Path(__file__).resolve().parents[2]  # project root
    paths = Paths.from_config(repo_root, cfg.raw)
    run_dir = run_ga(cfg, paths)
    typer.echo(f"✅ Finished. Results in: {run_dir}")

@app.command()
def clean_data(config: str = typer.Option("configs/default.yaml", help="Path to YAML config.")):
    _ = load_config(config)
    typer.echo("Data cleaning: implement by connecting your EDA script.")

@app.command()
def train(config: str = typer.Option("configs/default.yaml", help="Path to YAML config.")):
    _ = load_config(config)
    typer.echo("Training: implement by connecting your training script.")

if __name__ == "__main__":
    app()
