"""
rewind.py

Module containing function definitions for strategies to rewind weights during iterative magnitude pruning,

Author: Jordan Bourdeau
Date Created: 4/27/24    
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.harness import model as mod
from src.harness import utils

def rewind_model_weights(model: keras.Model, mask_model: keras.Model, rewind_rule: callable):
    """
    Function which rewinds a model's weights according to a specified rewind rule and the 
    masks in the `mask_model`.

    Args:
        model (keras.Model): Keras model to rewind the weights for.
        mask_model (keras.Model): Keras model containing weight masks.
        rewind_rule (callable): Callable which takes the model as input and resets
            weights to the specified rule. Does NOT apply masking.
    """
    rewind_rule(model)
    masked_weights: list[np.ndarray] = [weights * masks for weights, masks in zip(model.get_weights(), mask_model.get_weights())]
    model.set_weights(masked_weights)
    
def rewind_to_random_init(seed: int, initializer: tf.keras.initializers.Initializer, model: keras.Model):
    """
    Function which rewinds a model's weight initializations using a specified random seed and initializer.

    Args:
        seed (int): Random seed to use when reinitializing weights.
        initializer (tf.keras.initializers.Initializer): Initializer function to use which takes in a shape input.
        model (keras.Model): Keras model to initialize weights for.
    """
    utils.set_seed(seed)
    for layer in model.layers:
        if layer.trainable:
            weights_shape = [w.shape for w in layer.get_weights()]
            new_weights = [initializer(shape) for shape in weights_shape]
            layer.set_weights(new_weights)
            
def no_rewind(model: keras.Model):
    """
    Rewind rule which does not alter the model weights in any way.
    Used to continue training on masked model.

    Args:
        model (keras.Model): Keras model expected to be passed in.
    """
    pass

def rewind_to_original_init(seed: int, model: keras.Model):
    """
    Function which rewinds a model to its initial weights.

    Args:
        seed (int): Random seed a model was created on and its index in the `models/` directory.
        model (keras.Model): Keras model to rewind weights for.
    """
    original_model: keras.Model = mod.load_model(seed, initial=True)
    model.set_weights(original_model.get_weights())