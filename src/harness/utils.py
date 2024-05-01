"""
utils.py

File containing utility functions.
"""

from contextlib import contextmanager
import numpy as np
import os
import random
import sys
import tensorflow as tf
from tensorflow import keras

from src.harness import constants as C

def set_seed(random_seed: int):
    """
    Function to set random seed for reproducability.

    :param random_seed: Integer values for the random seed to set.
    """
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    
def is_prunable(layer: keras.layers.Layer) -> bool:
    """
    Function which checks if a given layer is prunable.

    Args:
        layer (keras.layers.Layer): Keras layer being examined.

    Returns:
        bool: True if the layer has weights, False otherwise.
    """
    prunable_types: list = [keras.layers.Conv2D, keras.layers.Conv1D, keras.layers.Dense]

    return any(isinstance(layer, t) for t in prunable_types)

def count_total_and_nonzero_params(model: keras.Model) -> tuple[int, int]:
    """
    Helper function to count the total number of parameters and number of nonzero parameters.

    :param model: Keras model to count the parameters for.

    :returns: Total number of weights and total number of nonzero weights.
    """
    weights: list[np.ndarray] = model.get_weights()
    total_weights: int = sum(tf.size(w).numpy() for w in weights)  # Calculate total weights
    nonzero_weights: int = sum(tf.math.count_nonzero(w).numpy() for w in weights)  # Calculate non-zero weights
    return total_weights, nonzero_weights

def count_total_and_nonzero_params_per_layer(model: keras.Model) -> list[tuple[int, int]]:
    """
    Helper function to count the total number of parameters and number of nonzero parameters
    for each layer.

    :param model: Keras model to count the parameters for.
`
    :returns: Total number of weights and total number of nonzero weights in each layer.
    """
    weights: list[np.ndarray] = model.get_weights()
    total_weights: list[int] = [tf.size(w).numpy() for w in weights] # Calculate total weights
    nonzero_weights: list[int] = [tf.math.count_nonzero(w).numpy() for w in weights]  # Calculate non-zero weights
    return list(zip(total_weights, nonzero_weights))

def get_layer_weight_counts(model: tf.keras.Model) -> list[int]:
    """
    Function to return a list of integer values for the number of 
    parameters in each layer.
    """
    def get_num_layer_weights(layer: tf.keras.layers.Layer) -> int:
        layer_weight_count: int = 0
        weights: list[np.array] = layer.get_weights()

        for idx in range(len(weights))[::2]:
            synapses: np.ndarray = weights[idx]
            biases: np.array = weights[idx + 1]
            layer_weight_count += np.prod(synapses.shape) + np.prod(biases.shape)

        return layer_weight_count
    
    return list(map(get_num_layer_weights, model.layers))

class SuppressOutput:
    """
    Context manager which suppresses all output generated within its contents.
    """
    def __enter__(self):
        self.devnull = open(os.devnull, 'w')
        self.old_stdout = sys.stdout
        sys.stdout = self.devnull
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.old_stdout
        self.devnull.close()