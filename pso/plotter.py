import matplotlib.pyplot as plt
import os
import numpy as np

class Plotter:
    """
    A class to handle all plotting functionalities.
    """
    def __init__(self, parameters, swarm, score_history, kinetic_energy_history, pbest_scores_history, lbest_scores_history):
        self.parameters = parameters
        self.swarm = swarm
        self.score_history = score_history
        self.kinetic_energy_history = kinetic_energy_history
        self.pbest_scores_history = pbest_scores_history
        self.lbest_scores_history = lbest_scores_history

    def save_all_plots(self, base_directory: str):
        """
        Saves all generated plots to the specified base directory.
        """
        # Save global performance plot
        self.save_global_plot(os.path.join(base_directory, 'pso_global_performance_symlog.png'))
        
        # Save particle fitness plots
        fitness_plots_dir = os.path.join(base_directory, 'particle_fitness_plots_symlog')
        self.save_particle_fitness_plots(fitness_plots_dir)
        
        # Save particle kinetic energy plots
        kinetic_energy_plots_dir = os.path.join(base_directory, 'particle_kinetic_energy_plots')
        self.save_particle_kinetic_energy_plots(kinetic_energy_plots_dir)

        # Save average kinetic energy plot
        self.save_kinetic_energy_plot(os.path.join(base_directory, 'pso_average_kinetic_energy_symlog.png'))
        
        # Save pbest vs lbest plots
        pbest_lbest_plots_dir = os.path.join(base_directory, 'particle_pbest_lbest_plots_symlog')
        self.save_pbest_lbest_plots(pbest_lbest_plots_dir)


    def save_global_plot(self, filename: str):
        """Saves a plot of the best score over iterations to a file."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.score_history, marker='o', linestyle='-', color='b')
        ax.set_title('PSO Best Score Over Iterations (Symlog Scale)')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Best Score')
        ax.grid(True)
        ax.set_yscale('symlog', linthresh=1e-5)
        fig.savefig(filename)
        plt.close(fig)
    
    def save_particle_fitness_plots(self, directory: str):
        """Saves a plot of each particle's fitness history to separate files."""
        os.makedirs(directory, exist_ok=True)
        num_particles = self.parameters['num_particles']
        
        for i in range(num_particles):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.swarm[i].energy_history, marker='o', linestyle='-')
            ax.set_title(f'Particle {i} Fitness History (Symlog Scale)')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Fitness (Score)')
            ax.grid(True)
            ax.set_yscale('symlog', linthresh=1e-5)
            
            filename = os.path.join(directory, f'particle_{i}_fitness.png')
            fig.savefig(filename)
            plt.close(fig)

    def save_particle_kinetic_energy_plots(self, directory: str):
        """Saves a plot of each particle's kinetic energy history to separate files."""
        os.makedirs(directory, exist_ok=True)
        num_particles = self.parameters['num_particles']
        
        for i in range(num_particles):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.swarm[i].kinetic_energy_history, marker='o', linestyle='-')
            ax.set_title(f'Particle {i} Kinetic Energy History (Symlog Scale)')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Kinetic Energy')
            ax.grid(True)
            ax.set_yscale('symlog', linthresh=1e-5)
            
            filename = os.path.join(directory, f'particle_{i}_kinetic_energy.png')
            fig.savefig(filename)
            plt.close(fig)

    def save_kinetic_energy_plot(self, filename: str):
        """Saves a plot of the average kinetic energy over iterations to a file."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.kinetic_energy_history, marker='o', linestyle='-', color='r')
        ax.set_title('Average Kinetic Energy Over Iterations (Symlog Scale)')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Average Kinetic Energy')
        ax.grid(True)
        ax.set_yscale('symlog', linthresh=1e-5)
        fig.savefig(filename)
        plt.close(fig)
        
    def save_pbest_lbest_plots(self, directory: str):
        """Saves plots of each particle's pbest and lbest scores to separate files."""
        os.makedirs(directory, exist_ok=True)
        num_particles = self.parameters['num_particles']
        
        for i in range(num_particles):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(self.pbest_scores_history[i], label='Personal Best (pbest)', linestyle='-')
            ax.plot(self.lbest_scores_history[i], label='Local Best (lbest)', linestyle='--')
            
            ax.set_title(f'Particle {i} Pbest vs Lbest Score (Symlog Scale)')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Score')
            ax.grid(True)
            ax.legend()
            ax.set_yscale('symlog', linthresh=1e-5)
            
            filename = os.path.join(directory, f'particle_{i}_pbest_lbest.png')
            fig.savefig(filename)
            plt.close(fig)
