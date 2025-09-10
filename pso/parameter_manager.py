import yaml
import os

class ParameterManager:
    """
    Manages the saving and loading of PSO parameters to and from YAML files.
    """
    @staticmethod
    def save_parameters(filepath: str, params: dict):
        """
        Saves a dictionary of parameters to a YAML file.

        Args:
            filepath (str): The full path to the output YAML file.
            params (dict): The dictionary of parameters to save.
        """
        try:
            with open(filepath, 'w') as f:
                yaml.safe_dump(params, f, sort_keys=False)
            print(f"Parameters successfully saved to {filepath}")
        except Exception as e:
            print(f"Error saving parameters to {filepath}: {e}")

    @staticmethod
    def load_parameters(filepath: str) -> dict:
        """
        Loads a dictionary of parameters from a YAML file.

        Args:
            filepath (str): The full path to the input YAML file.

        Returns:
            dict: The loaded parameters.
        
        Raises:
            FileNotFoundError: If the specified file does not exist.
            yaml.YAMLError: If there is a syntax error in the YAML file.
            KeyError: If essential keys are missing from the loaded parameters.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Parameter file not found: {filepath}")

        with open(filepath, 'r') as f:
            try:
                params = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                raise yaml.YAMLError(f"YAML syntax error in file {filepath}: {exc}") from exc
        
        # Proper checks should be added here to ensure all expected keys exist.
        # This is a basic example; a real-world implementation might be more extensive.
        required_keys = ['dimensions', 'num_particles', 'max_iterations', 'w', 'c1', 'c2', 'neighborhood_size', 
                         'position_bounds', 'velocity_bounds', 'noise_std_dev', 'dt']
        for key in required_keys:
            if key not in params:
                raise KeyError(f"Missing required key '{key}' in parameter file: {filepath}")

        return params
