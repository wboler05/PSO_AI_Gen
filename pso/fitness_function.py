from abc import ABC, abstractmethod
import numpy as np

class AbstractFitnessFunction(ABC):
    """
    Abstract base class for defining a maximization fitness function.
    Inherit from this class and implement the 'evaluate' method.
    """
    @abstractmethod
    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluate the fitness of a given particle's position.
        The function should return a higher value for better solutions.
        """
        pass
