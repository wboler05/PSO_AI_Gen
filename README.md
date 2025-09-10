# Particle Swarm Optimization (PSO) Library

## Overview

This project provides a modular and extensible implementation of the Particle Swarm Optimization (PSO) algorithm in Python. It is designed to be a clear, commented, and educational example of how to implement a metaheuristic optimization technique. The library features a Local-Best PSO variant, configurable parameters, support for custom fitness functions, and comprehensive plotting of results to aid in analysis.

The implementation is structured for clarity, with separate modules for the core algorithm, the particles themselves, the fitness functions, parameter management, and result visualization. Example scripts are provided for classic optimization problems to demonstrate usage.

## Setup and Installation

### Prerequisites

*   Python 3.10
*   A Conda environment manager

### Installation

1.  **Clone the repository**:
    ```sh
    git clone https://github.com/your-username/pso-library.git
    cd pso-library
    ```
2.  **Create and activate the Conda environment**:
    Using the provided `env.yaml` file, you can create a dedicated environment with all necessary dependencies.
    ```sh
    conda env create -f env.yaml
    conda activate pso_env
    ```

## Project Structure


└── .gitignore
└── env.yaml
├── examples/
│ ├── run_rastrigin_maximization.py
│ ├── run_rosenbrock_maximization.py
│ └── run_sphere_maximization.py
├── pso/
│ ├── init.py
│ ├── fitness_function.py
│ ├── parameter_manager.py
│ ├── particle.py
│ ├── plotter.py
│ └── pso_algorithm.py
├── tests/
│ └── test_pso_components.py
└── README.md

## Code Description

### `pso/pso_algorithm.py`
This is the core of the PSO implementation. It defines the `LocalBestPSO` class, which manages the entire optimization process.
*   **Initialization**: Can be configured via a Python dictionary or a YAML file.
*   **Optimization Loop**: Iterates through the swarm, evaluates particle fitness, and updates velocities and positions.
*   **Neighborhood**: Implements a local-best topology (ring) for social influence.
*   **Result Handling**: Calls the `ParameterManager` and `Plotter` to save results and generate visualizations.

### `pso/particle.py`
The `Particle` class represents a single candidate solution in the swarm.
*   **State**: Tracks its current `position`, `velocity`, `best_position` found so far, and `best_score`.
*   **Updates**: Includes methods for updating its velocity (based on inertia, cognitive, social, and noise factors) and position.
*   **Metrics**: Stores a history of its fitness scores (`energy_history`) and kinetic energy.

### `pso/fitness_function.py`
This file defines the abstract base class `AbstractFitnessFunction`.
*   **`evaluate(self, position: np.ndarray)`**: An abstract method that must be implemented by concrete fitness functions.
*   **Purpose**: Ensures all fitness functions conform to a standard interface, making it easy to swap optimization problems.

### `pso/parameter_manager.py`
A utility class for managing simulation parameters.
*   **`save_parameters(filepath, params)`**: Saves a parameter dictionary to a YAML file.
*   **`load_parameters(filepath)`**: Loads a parameter dictionary from a YAML file, with validation checks for required keys.

### `pso/plotter.py`
Handles all visualization aspects of the simulation.
*   **`save_all_plots(base_directory)`**: A top-level method to save a variety of plots to a specified directory.
*   **Plots**: Includes visualizations for global best score, average kinetic energy, and individual particle histories. Uses a symmetric logarithmic scale for improved data representation.

### `examples/`
This directory contains example scripts demonstrating how to use the PSO library.
*   `run_sphere_maximization.py`: A simple, unimodal maximization problem.
*   `run_rastrigin_maximization.py`: A corrected version of the Rastrigin function, a challenging multimodal problem.
*   `run_rosenbrock_maximization.py`: Another classic and challenging optimization problem involving navigating a narrow valley.

## Usage

To run the examples correctly using the `-m` module method, you must execute the commands from the project's root directory (`pso-library/`).

1.  **Navigate to the project root directory**:
    ```sh
    cd pso-library
    ```
2.  **Run one of the example scripts as a module:**
    ```sh
    python -m examples.run_sphere_maximization
    ```
    or
    ```sh
    python -m examples.run_rastrigin_maximization
    ```
    or
    ```sh
    python -m examples.run_rosenbrock_maximization
    ```
3.  **Check the output:** The script will print progress to the console during optimization and then save a `pso_results` directory containing the final parameters, a text file with the best solution, and various plots.

## Running Tests

Unit tests are provided to ensure the core components of the PSO algorithm and supporting modules function as expected.

1.  **Activate the Conda environment**:
    ```sh
    conda activate pso_env
    ```
2.  **Run the tests from the project root directory**:
    ```sh
    pytest
    ```
    `pytest` will automatically discover and run the tests in the `tests/` directory.

## Customizing the Simulation

To run your own optimization problem, you can:
1.  Create a new Python file in the `examples` directory.
2.  Define a new class that inherits from `AbstractFitnessFunction` and implements the `evaluate` method.
3.  Instantiate your new fitness function and the `LocalBestPSO` class.
4.  Modify the `pso_parameters` dictionary to suit your specific problem (e.g., number of dimensions, bounds).
5.  Run your script using the `python -m examples.<your_script_name>` command from the project root.

## AI Assistance

This code was developed with the assistance of a large language model (LLM) trained by Google. The LLM was used for tasks such as structuring the project, generating boilerplate code, providing explanations, and refining existing code snippets based on provided requirements.

Please note that while the LLM was a helpful tool, the final code was reviewed and validated to ensure it meets the specified functionality and standards.


