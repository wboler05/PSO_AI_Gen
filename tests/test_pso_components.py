import pytest
import numpy as np
import numpy.testing as npt
import os
import shutil
import yaml

from pso.pso_algorithm import LocalBestPSO
from pso.particle import Particle
from pso.fitness_function import AbstractFitnessFunction
from pso.parameter_manager import ParameterManager

# Mock fitness function for testing purposes
class MockFitnessFunction(AbstractFitnessFunction):
    def evaluate(self, position: np.ndarray) -> float:
        # A simple function for testing, e.g., Sphere maximization
        return -np.sum(position**2)

# Fixture for common PSO parameters
@pytest.fixture
def pso_params():
    return {
        'dimensions': 2,
        'num_particles': 10,
        'max_iterations': 5,
        'w': 0.7,
        'c1': 1.5,
        'c2': 1.5,
        'neighborhood_size': 2,
        'position_bounds': (-5.0, 5.0),
        'velocity_bounds': (-0.5, 0.5),
        'noise_std_dev': 0.05,
        'dt': 1.0,
    }

# Helper function for robust dictionary comparison
def compare_params(dict1, dict2):
    """
    Compares two parameter dictionaries, tolerating minor float inaccuracies and
    type differences between lists and tuples for sequences.
    """
    if dict1.keys() != dict2.keys():
        return False
    
    for key in dict1:
        val1 = dict1[key]
        val2 = dict2[key]
        
        # Use a relative tolerance for floating-point comparisons
        if isinstance(val1, float) and isinstance(val2, float):
            if not np.isclose(val1, val2):
                return False
        # Handle sequences (lists and tuples) by converting to a common type
        elif isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
            if len(val1) != len(val2):
                return False
            for v1, v2 in zip(val1, val2):
                if not np.isclose(v1, v2):
                    return False
        # Fallback to direct comparison for other types
        elif val1 != val2:
            return False
            
    return True

# --- Test `pso/particle.py` ---

def test_particle_initialization(pso_params):
    particle = Particle(pso_params)
    assert len(particle.position) == pso_params['dimensions']
    assert np.all(particle.velocity == 0.0)
    assert np.all(np.isclose(particle.best_position, particle.position))
    assert particle.best_score == float('-inf')

def test_particle_update_velocity(pso_params):
    np.random.seed(42)  # For deterministic randomness
    particle = Particle(pso_params)
    local_best_position = np.array([1.0, 1.0])
    
    initial_velocity = np.copy(particle.velocity)
    particle.update_velocity(local_best_position)
    
    # Velocity should have changed
    assert not np.all(np.isclose(particle.velocity, initial_velocity))
    
    # Velocity should be within bounds
    min_vel, max_vel = pso_params['velocity_bounds']
    assert np.all(particle.velocity >= min_vel)
    assert np.all(particle.velocity <= max_vel)

def test_particle_update_position(pso_params):
    particle = Particle(pso_params)
    
    # Set a known initial position to avoid random float issues
    initial_position = np.array([1.0, 2.0])
    particle.position = np.copy(initial_position)
    
    # Set a known velocity for predictable position change
    particle.velocity = np.array([0.1, 0.2])
    particle.update_position()
    
    expected_position = initial_position + particle.velocity * pso_params['dt']
    
    # Use numpy.testing.assert_allclose for more reliable comparison
    npt.assert_allclose(particle.position, expected_position)

def test_particle_position_clamping(pso_params):
    pso_params['position_bounds'] = (0, 1)
    particle = Particle(pso_params)
    
    # Set position outside of bounds and large velocity to test clamping
    particle.position = np.array([0.9, 0.9])
    particle.velocity = np.array([0.5, 0.5])
    particle.update_position()
    
    # Position should be clamped back within bounds
    assert np.all(particle.position <= 1)
    assert np.all(particle.position >= 0)

# --- Test `pso/parameter_manager.py` ---

def test_save_and_load_parameters(tmp_path, pso_params):
    filepath = tmp_path / "test_params.yaml"
    ParameterManager.save_parameters(str(filepath), pso_params)
    
    loaded_params = ParameterManager.load_parameters(str(filepath))
    
    assert compare_params(loaded_params, pso_params)

def test_load_parameters_file_not_found():
    with pytest.raises(FileNotFoundError):
        ParameterManager.load_parameters("non_existent_file.yaml")

def test_load_parameters_missing_key(tmp_path):
    filepath = tmp_path / "incomplete_params.yaml"
    incomplete_params = {'dimensions': 2, 'num_particles': 10}
    with open(filepath, 'w') as f:
        yaml.safe_dump(incomplete_params, f)
        
    with pytest.raises(KeyError):
        ParameterManager.load_parameters(str(filepath))

# --- Test `pso/pso_algorithm.py` ---

def test_pso_initialization(pso_params):
    fitness_func = MockFitnessFunction()
    pso = LocalBestPSO(fitness_function=fitness_func, parameters=pso_params)
    
    assert compare_params(pso.parameters, pso_params)
    assert len(pso.swarm) == pso_params['num_particles']
    assert pso.global_best_score == float('-inf')

def test_pso_initialization_with_yaml(tmp_path, pso_params):
    filepath = tmp_path / "test_params.yaml"
    ParameterManager.save_parameters(str(filepath), pso_params)
    
    fitness_func = MockFitnessFunction()
    pso = LocalBestPSO(fitness_function=fitness_func, yaml_filepath=str(filepath))
    
    assert compare_params(pso.parameters, pso_params)

def test_pso_optimization_progress(pso_params):
    fitness_func = MockFitnessFunction()
    pso = LocalBestPSO(fitness_function=fitness_func, parameters=pso_params)
    
    initial_best_score = pso.global_best_score
    pso.optimize()
    
    # Best score should have improved after optimization
    assert pso.global_best_score > initial_best_score
    # History should have been recorded
    assert len(pso.score_history) == pso_params['max_iterations']

def test_pso_neighborhood_best_is_found(pso_params):
    pso_params['num_particles'] = 5
    pso_params['neighborhood_size'] = 1
    fitness_func = MockFitnessFunction()
    pso = LocalBestPSO(fitness_function=fitness_func, parameters=pso_params)

    # Manually set best scores for particles to test neighborhood logic
    pso.swarm[0].best_score = -50
    pso.swarm[1].best_score = -10
    pso.swarm[2].best_score = -100
    pso.swarm[3].best_score = -5
    pso.swarm[4].best_score = -200

    # For particle 2, its neighborhood is particles 1, 2, and 3
    # Scores are [-10, -100, -5]
    lbest_score = pso._get_local_best_score(2)
    expected_lbest_score = -5
    assert lbest_score == expected_lbest_score

def test_pso_save_results(tmp_path, pso_params):
    fitness_func = MockFitnessFunction()
    pso = LocalBestPSO(fitness_function=fitness_func, parameters=pso_params)
    pso.optimize()
    
    output_dir = str(tmp_path / "test_results")
    pso.save_results(output_dir)
    
    assert os.path.exists(output_dir)
    assert os.path.exists(os.path.join(output_dir, 'parameters.yaml'))
    assert os.path.exists(os.path.join(output_dir, 'results.txt'))
    assert os.path.exists(os.path.join(output_dir, 'pso_global_performance_symlog.png'))

    # Cleanup after test
    shutil.rmtree(output_dir)


# --- Test `pso/fitness_function.py` ---

def test_abstract_fitness_function():
    with pytest.raises(TypeError):
        # Cannot instantiate an abstract class
        AbstractFitnessFunction()
    
    class ConcreteFitness(AbstractFitnessFunction):
        def evaluate(self, position: np.ndarray) -> float:
            return 1.0
    
    # Can instantiate a concrete subclass
    assert ConcreteFitness().evaluate(np.array([])) == 1.0
