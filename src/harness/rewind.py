"""
rewind.py

Module containing function definitions for strategies to rewind weights during iterative magnitude pruning.

Interface specs:

    - The main function is `rewind_model_weights` which takes a model, its masks, a rewind rule,
    and any positional arguments to pass into the rewind rule. This function will rewind the
    model weights according to the strategy in the `rewind_rule` function and then mask the
    weights according to its model masks.
    
    - Rewind rules only have to be able to take a single parameter which is the keras model
    being rewound, and can take any number of positional arguments.

Author: Jordan Bourdeau
Date Created: 4/27/24    
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.harness import model as mod
from src.harness import utils

def rewind_model_weights(
    model: keras.Model, 
    mask_model: keras.Model, 
    rewind_rule: callable,
    *rewind_args,
    ):
    """
    Function which rewinds a model's weights according to a specified rewind rule and the 
    masks in the `mask_model`.

    Args:
        model (keras.Model): Keras model to rewind the weights for.
        mask_model (keras.Model): Keras model containing weight masks.
        rewind_rule (callable): Callable which takes the model as input and resets
            weights to the specified rule.
        *rewind_args: Arguments to pass into the rewind rule.
    """
    rewind_rule(model, *rewind_args)
    masked_weights = [weights * masks for weights, masks in zip(model.get_weights(), mask_model.get_weights())]
    model.set_weights(masked_weights)
    
def rewind_to_random_init(
    model: keras.Model,
    seed: int, 
    initializer: tf.keras.initializers.Initializer, 
    ):
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
            
def no_rewind(*args):
    """
    Rewind rule which does not alter the model weights in any way.
    Used to continue training on masked model.

    Args:
        *args: Arguments which get ignored.
    """
    pass

def get_rewind_to_original_init_for(seed: int, directory: str = './') -> callable:
    """
    Function which returns a rewind rule that returns to original initialization
    with a directory instantiated in it.

    Args:
        directory (str, optional): Directory to look for weights to rewind. Defaults to './'.

    Returns:
        callable: Rewind rule for a specific directory.
    """

    def rewind_to_original_init(model: keras.Model):
        """
        Function which rewinds a model to its initial weights.

        Args:
            model (keras.Model): Keras model to rewind weights for.
        """
        original_model = mod.load_model(seed, initial=True, directory=directory)
        model.set_weights(original_model.get_weights())
        
    return rewind_to_original_init
