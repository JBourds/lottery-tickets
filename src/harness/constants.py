"""
constants.py

File containing all constants and constant functions.
"""

import functools
import tensorflow as tf

# File Locations
DATA_DIRECTORY: str = 'data/'
MODEL_DIRECTORY: str = 'models/'
CHECKPOINT_DIRECTORY: str = 'checkpoints/'
FIT_DIRECTORY: str = 'logs/fit/'

DIRECTORIES: list[str] = [
    MODEL_DIRECTORY,
    CHECKPOINT_DIRECTORY,
    FIT_DIRECTORY,
    DATA_DIRECTORY,
]

# Training Parameters
MNIST_LOCATION: str = DATA_DIRECTORY + 'mnist/'
TRAINING_ITERATIONS: int = 50_000
OPTIMIZER = functools.partial(tf.keras.optimizers.SGD, .1)