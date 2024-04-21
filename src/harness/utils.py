"""
utils.py

File containing utility functions.
"""

import numpy as np
import os
import random
import tensorflow as tf
from tensorflow import keras

from src.harness.constants import Constants as C

def set_seed(random_seed: int):
    """
    Function to set random seed for reproducability.

    :param random_seed: Integer values for the random seed to set.
    """
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

def count_params(model: keras.Model) -> tuple[int, int]:
    """
    Helper function to count the total number of parameters and number of nonzero parameters.
    """
    weights = model.get_weights()
    total_weights = sum(tf.size(w).numpy() for w in weights)  # Calculate total weights
    nonzero_weights = sum(tf.math.count_nonzero(w).numpy() for w in weights)  # Calculate non-zero weights
    return total_weights, nonzero_weights