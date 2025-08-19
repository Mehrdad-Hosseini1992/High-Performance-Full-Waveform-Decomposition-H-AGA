# optimizer.py
# Defines the base Optimizer class using abc (Abstract Base Class).
# This establishes a contract that all optimizer subclasses must follow,
# ensuring they have an 'optimize' method. This is a good use of
# inheritance and polymorphism as requested.

from abc import ABC, abstractmethod
import numpy as np

class Optimizer(ABC):
    """
    Abstract base class for optimization strategies.
    Defines the common interface for all optimizers.
    """
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray, initial_params: list):
        """
        Initializes the Optimizer.

        Args:
            x_data (np.ndarray): The time data.
            y_data (np.ndarray): The amplitude data.
            initial_params (list): A list of dictionaries with initial parameter guesses.
        """
        self.x_data = x_data
        self.y_data = y_data
        self.initial_params = initial_params
        self.num_components = len(initial_params)

    @abstractmethod
    def optimize(self):
        """
        The core method to be implemented by all subclasses.
        This method should run the specific optimization algorithm.
        """
        pass

