"""
constants.py

File containing all constants and constant functions.
"""

import functools
import tensorflow as tf
from sys import platform

# File Locations
DATA_DIRECTORY: str = 'data'
MODEL_DIRECTORY: str = 'models'

DIRECTORIES: list[str] = [
    MODEL_DIRECTORY,
    DATA_DIRECTORY,
]

# Training Parameters

PATIENCE: int = 3
MINIMUM_DELTA: float = 0.0001
LEARNING_RATE: float = 0.005
if platform.lower() == 'darwin':
    OOPTIMIZER = functools.partial(tf.keras.optimizers.legacy.Adam, LEARNING_RATE)
else:
    OPTIMIZER = functools.partial(tf.keras.optimizers.Adam, LEARNING_RATE)

LOSS_FUNCTION: tf.keras.losses.Loss = functools.partial(tf.keras.losses.CategoricalCrossentropy)

# Test Experiment Parameters
TEST_NUM_MODELS: int = 2
TEST_TRAINING_EPOCHS: int = 2
TEST_PRUNING_STEPS: int = 2

# Real Experiment Parameters
NUM_MODELS: int = 100
TRAINING_EPOCHS: int = 60
BATCH_SIZE: int = 128
PRUNING_STEPS: int = 45
