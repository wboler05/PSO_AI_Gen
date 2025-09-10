import numpy as np
import os
from pso.particle import Particle
from pso.fitness_function import AbstractFitnessFunction
from pso.parameter_manager import ParameterManager
from pso.plotter import Plotter

class LocalBestPSO:
    """
    The Local-Best Particle Swarm Optimization algorithm with a fixed neighborhood size.
    This version includes bounds on position and velocity and tracks particle energy.
    All parameters are stored in a shared dictionary.
    """
    def __init__(self, fitness_function: AbstractFitnessFunction, parameters: dict = None, yaml_filepath: str = None):
        """
        Initialize the Local-Best PSO algorithm.

        Args:
            fitness_function (AbstractFitnessFunction): The function to maximize.
            parameters (dict): A dictionary containing all simulation parameters.
            yaml_filepath (str): The path to a YAML file to load parameters from.
        """
        if yaml_filepath:
            self.parameters = ParameterManager.load_parameters(yaml_filepath)
        elif parameters:
            self.parameters = parameters
        else:
            raise ValueError("Either a 'parameters' dictionary or a 'yaml_filepath' must be provided.")

        self.fitness_function = fitness_function

        if not (1 <= self.parameters['neighborhood_size'] < self.parameters['num_particles'] / 2):
            raise ValueError("Neighborhood size must be between 1 and num_particles/2 - 1.")

        self.swarm = [Particle(self.parameters) for _ in range(self.parameters['num_particles'])]
        self.global_best_position = None
        self.global_best_score = float('-inf')
        self.score_history = []
        self.kinetic_energy_history = []
        self.pbest_scores_history = [[] for _ in range(self.parameters['num_particles'])]
        self.lbest_scores_history = [[] for _ in range(self.parameters['num_particles'])]

    def save_parameters(self, filepath: str):
        """
        Saves the PSO parameters to a YAML file.
        """
        ParameterManager.save_parameters(filepath, self.parameters)

    def _get_local_best_position(self, particle_index: int) -> np.ndarray:
        """
        Find the best position within a particle's local neighborhood (ring topology).
        """
        best_candidate = self.swarm[particle_index]
        neighborhood_size = self.parameters['neighborhood_size']
        num_particles = self.parameters['num_particles']
        
        for offset in range(-neighborhood_size, neighborhood_size + 1):
            if offset == 0:
                continue
            neighbor_index = (particle_index + offset + num_particles) % num_particles
            neighbor = self.swarm[neighbor_index]
            if neighbor.best_score > best_candidate.best_score:
                best_candidate = neighbor
        return best_candidate.best_position

    def _get_local_best_score(self, particle_index: int) -> float:
        """
        Find the best score within a particle's local neighborhood (ring topology).
        """
        best_candidate = self.swarm[particle_index]
        neighborhood_size = self.parameters['neighborhood_size']
        num_particles = self.parameters['num_particles']
        
        for offset in range(-neighborhood_size, neighborhood_size + 1):
            if offset == 0:
                continue
            neighbor_index = (particle_index + offset + num_particles) % num_particles
            neighbor = self.swarm[neighbor_index]
            if neighbor.best_score > best_candidate.best_score:
                best_candidate = neighbor
        return best_candidate.best_score

    def optimize(self) -> tuple:
        """
        Run the Local-Best PSO algorithm.

        Returns:
            tuple: A tuple containing the overall best position and score.
        """
        max_iterations = self.parameters['max_iterations']
        for iteration in range(max_iterations):
            total_kinetic_energy = 0
            
            # Phase 1: Evaluate fitness, update pbest, and find lbest for the current iteration
            for i, particle in enumerate(self.swarm):
                current_score = self.fitness_function.evaluate(particle.position)
                particle.energy_history.append(current_score)
                if current_score > particle.best_score:
                    particle.best_score = current_score
                    particle.best_position = np.copy(particle.position)
                
                self.pbest_scores_history[i].append(particle.best_score)
                lbest_score = self._get_local_best_score(i)
                self.lbest_scores_history[i].append(lbest_score)

            # Phase 2: Update velocity and position, and calculate total kinetic energy
            for i, particle in enumerate(self.swarm):
                local_best_position = self._get_local_best_position(i)
                particle.update_velocity(local_best_position)
                particle.update_position()
                total_kinetic_energy += particle.kinetic_energy
            
            average_kinetic_energy = total_kinetic_energy / self.parameters['num_particles']
            self.kinetic_energy_history.append(average_kinetic_energy)
            
            # Update global best and record global score
            for particle in self.swarm:
                if particle.best_score > self.global_best_score:
                    self.global_best_score = particle.best_score
                    self.global_best_position = np.copy(particle.best_position)
            self.score_history.append(self.global_best_score)
            
            print(f"Iteration {iteration+1}/{max_iterations}: Overall best score = {self.global_best_score:.4f}, Avg Kinetic Energy = {average_kinetic_energy:.4f}")

        return self.global_best_position, self.global_best_score

    def save_results(self, output_dir: str):
        """
        Encapsulates the entire process of saving all results.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save parameters
        params_filename = os.path.join(output_dir, 'parameters.yaml')
        self.save_parameters(params_filename)
        
        # Save solution
        results_filename = os.path.join(output_dir, 'results.txt')
        with open(results_filename, 'w') as f:
            f.write("--- Optimization Results ---\n")
            f.write(f"Best Position: {self.global_best_position}\n")
            f.write(f"Best Score: {self.global_best_score}\n")
        
        # Initialize plotter and save all plots
        plotter = Plotter(self.parameters, self.swarm, self.score_history, self.kinetic_energy_history, self.pbest_scores_history, self.lbest_scores_history)
        plotter.save_all_plots(output_dir)

        print(f"Results saved to: {results_filename}")
        print(f"Parameters saved to: {params_filename}")
        print(f"Plots saved to directory: {output_dir}")

