"""
constants.py

File containing all constants and constant functions.
"""

from enum import Enum
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

PATIENCE: int = 3
MINIMUM_DELTA: float = 0.0001
LEARNING_RATE: float = 0.005
OPTIMIZER = functools.partial(tf.keras.optimizers.legacy.Adam, LEARNING_RATE)
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

class OriginalParams(Enum):
    """
    Class acting as a namespace for original LTH paper parameters
    by Frankle and Carbin.
    
    Taken largely from figure 2.
    """
    
    LEARNING_RATE: float = 0.0012
    if platform == 'darwin':
        OPTIMIZER: tf.keras.optimizers.Optimizer = functools.partial(tf.keras.optimizers.legacy.Adam, LEARNING_RATE)
    else:
        OPTIMIZER: tf.keras.optimizers.Optimizer = functools.partial(tf.keras.optimizers.Adam, LEARNING_RATE)
        
    LENET_BATCH_SIZE: int = 60
    RESNET_BATCH_SIZE: int = 128
    VGG_BATCH_SIZE: int = 64
    
    CONV_PRUNING_RATE: float = 0.1
    FC_PRUNING_RATE: float = 0.2
    # The number of iterations (batches) to train for at a time before evaluating
    # validation set performance and checking for early stopping
    PERFORMANCE_EVALUATION_FREQUENCY: int = 100
