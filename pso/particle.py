import numpy as np
import random

class Particle:
    """
    Represents a single particle in the swarm.
    """
    def __init__(self, parameters: dict):
        """
        Initialize the particle with a random position and zero velocity.

        Args:
            parameters (dict): A shared dictionary containing all simulation parameters.
        """
        self.parameters = parameters
        dimensions = self.parameters['dimensions']
        position_bounds = self.parameters['position_bounds']
        
        min_pos_bound, max_pos_bound = position_bounds
        self.position = np.array([random.uniform(min_pos_bound, max_pos_bound) for _ in range(dimensions)])
        
        self.velocity = np.zeros(dimensions)
        self.kinetic_energy = 0.0 # New member for kinetic energy
        
        self.best_position = np.copy(self.position)
        self.best_score = float('-inf')
        
        self.energy_history = []  # Tracks fitness score (positional energy)
        self.kinetic_energy_history = [] # New member for kinetic energy history

    def update_velocity(self, local_best_position: np.ndarray):
        """
        Update the particle's velocity based on its own best position,
        the swarm's local best position, inertia, and Gaussian noise.
        """
        w = self.parameters['w']
        c1 = self.parameters['c1']
        c2 = self.parameters['c2']
        noise_std_dev = self.parameters['noise_std_dev']
        velocity_bounds = self.parameters['velocity_bounds']
        dimensions = self.parameters['dimensions']
        
        r1 = random.uniform(0, 1)
        r2 = random.uniform(0, 1)

        cognitive_velocity = c1 * r1 * (self.best_position - self.position)
        social_velocity = c2 * r2 * (local_best_position - self.position)
        
        gaussian_noise = np.random.normal(0, noise_std_dev, dimensions)
        
        self.velocity = (w * self.velocity) + cognitive_velocity + social_velocity + gaussian_noise
        
        min_vel_bound, max_vel_bound = velocity_bounds
        self.velocity = np.clip(self.velocity, min_vel_bound, max_vel_bound)

        # Calculate and store kinetic energy after velocity is finalized
        speed_squared = np.sum(self.velocity**2)
        self.kinetic_energy = 0.5 * speed_squared
        self.kinetic_energy_history.append(self.kinetic_energy)

    def update_position(self):
        """
        Update the particle's position by adding its velocity.
        Clamps the position to stay within the defined bounds.
        """
        dt = self.parameters['dt']
        position_bounds = self.parameters['position_bounds']
        
        min_pos_bound, max_pos_bound = position_bounds
        self.position += self.velocity * dt
        self.position = np.clip(self.position, min_pos_bound, max_pos_bound)

