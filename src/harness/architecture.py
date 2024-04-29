"""
architecture.py

Module containing architectural definitions for networks being tested.
NOTE: Not currently in use as only one architecture is being used at the moment.

Author: Jordan Bourdeau
Date Created: 4/28/24
"""

from enum import Enum
from tensorflow import keras
import tensorflow as tf

class Architecture:
    """
    Base class representing a given architecture testing is being performed on.
    """
    def make_model(self) -> keras.Model:
        raise NotImplementedError()
    
    def get_pruning_rate(self) -> float:
        raise NotImplementedError()
    
    def get_optimizer(self) -> tf.optimizers.Optimizer:
        raise NotImplementedError()
    
    def get_iterations(self) -> int:
        raise NotImplementedError()
    
    def get_num_batches(self) -> int:
        raise NotImplementedError()
    
    def get_batch_size(self) -> int:
        raise NotImplementedError()
    
    def get_total_weights(self) -> int:
        raise NotImplementedError()
    
    