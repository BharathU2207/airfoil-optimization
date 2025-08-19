# Machine Learning-Assisted Airfoil Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

This repository contains the official implementation for the paper: **"Machine Learning Driven Optimisation of Low-Reynolds Number UAV Airfoils Using Genetic Algorithm"**.

This project provides a data-driven framework for the aerodynamic optimization of low-Reynolds number airfoils, primarily aimed at small UAV applications. It integrates a machine learning surrogate model with a multi-objective genetic algorithm (GA) to rapidly generate and refine high-performance airfoil shapes.

## Key Features

* **Airfoil Parameterization**: Uses the Class-Shape Transformation (CST) method to define airfoil geometries with a minimal set of parameters.
* **Surrogate Modeling**: Employs a trained XGBoost model to provide high-speed, accurate predictions of aerodynamic efficiency ($l/d$), replacing computationally expensive CFD simulations during optimization.
* **Multi-Objective Optimization**: Leverages the NSGA-II genetic algorithm to explore the design space, balancing multiple objectives and geometric constraints to produce robust airfoil designs.
* **Reproducible & Configurable**: All parameters are managed through a central configuration file, and results from each run are saved to a unique, timestamped directory for full reproducibility.

## Installation

To get started, clone the repository and install the required dependencies. It is highly recommended to use a virtual environment.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/airfoil-opt.git](https://github.com/your-username/airfoil-opt.git)
    cd airfoil-opt
    ```

2.  **Create and activate a virtual environment (optional but recommended):**
    ```bash
    # For Windows
    python -m venv airfoil_env
    .\airfoil_env\Scripts\activate

    # For macOS/Linux
    python3 -m venv airfoil_env
    source airfoil_env/bin/activate
    ```

3.  **Install the project in editable mode:**
    This command will install all required dependencies and make the `airfoil-opt` command-line tool available in your terminal.
    ```bash
    pip install -e .
    ```

## Usage

The primary way to use this tool is through the command-line interface.

### Quick Start

To run an optimization using the default settings specified in `configs/default.yaml`:

```bash
airfoil-opt optimize
```

### Overriding Parameters

You can easily override key parameters from the command line for quick experiments. This is useful for running tests without modifying the configuration file.

For example, to run a short test with a smaller population and fewer generations:

```bash
airfoil-opt optimize --population 50 --generations 10
```

## Output

After each run, a new directory is created in the `runs/` folder with a timestamp (e.g., `runs/20250818-093000/`). This directory contains:

* **`meta.json`**: A complete record of the configuration used for the run and a history of key metrics for each generation.
* **`pareto.csv`**: The fitness values and CST coefficients of the final Pareto-optimal solutions.
* **`plots/`**: A folder containing several informative plots, including:
    * `pareto_front.png`: The final Pareto front.
    * `optimized_vs_baseline_shape.png`: A comparison of the final optimized airfoil shape against the baseline.
    * `f1_over_gens.png` & `f2_over_gens.png`: The evolution of the best and average scores for each objective over the generations.
    * `airfoil_evolution.png`: Snapshots of the best airfoil shape at different stages of the evolution.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.