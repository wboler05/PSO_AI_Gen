import numpy as np
import random
import os
import sys
from abc import ABC, abstractmethod
import datetime

# Add the parent directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import PSO components from the library
from pso.pso_algorithm import LocalBestPSO
from pso.fitness_function import AbstractFitnessFunction


# Concrete Example of a Multimodal Maximization Fitness Function
class RastriginMaximization(AbstractFitnessFunction):
    def evaluate(self, position: np.ndarray) -> float:
        n = len(position)
        return -(10 * n + np.sum(position**2 - 10 * np.cos(2 * np.pi * position)))

def main():
    """
    Main function to run the PSO simulation for Rastrigin Maximization.
    """
    pso_parameters = {
        'dimensions': 2,
        'num_particles': 100,
        'max_iterations': 500,
        'w': 0.7,
        'c1': 1.5,
        'c2': 1.5,
        'neighborhood_size': 3,
        'position_bounds': (-5.12, 5.12),
        'velocity_bounds': (-0.5, 0.5),
        'noise_std_dev': 0.05,
        'dt': 1.0,
    }
    
    fitness_func = RastriginMaximization()
    pso = LocalBestPSO(fitness_function=fitness_func, parameters=pso_parameters)
    
    best_position, best_score = pso.optimize()
    
    # Example script now responsible for creating and passing the output directory path
    example_name, _ = os.path.splitext(os.path.basename(__file__))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join('pso_results', example_name, timestamp)

    pso.save_results(output_dir)

    print("\n--- Optimization Complete ---")
    print(f"Overall Best Position: {best_position}")
    print(f"Overall Best Score: {best_score}")
    print("\n--- Particle Energy History ---")
    for i in range(min(5, pso_parameters['num_particles'])):
        print(f"Particle {i} fitness history: {pso.swarm[i].energy_history[:10]}...")

if __name__ == "__main__":
    main()
