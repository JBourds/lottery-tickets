"""
model.py

Module containing definitions for functions/classes used to create/load/save models.

Authors: Jordan Bourdeau, Casey Forey
Data Created: 3/8/24
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Flatten, Input

from src.harness import constants as C
from src.harness import paths

def load_model(seed: int, pruning_step: int = 0, masks: bool = False, initial: bool = False) -> tf.keras.Model:
    """
    Function used to load a single trained model.

    :param seed:           Random seed the model was trained using
    :param pruning_step:   Integer value for the number of pruning steps which had been completed for the model.
    :param masks:          Boolean for whether the model masks are being retrieved or not.
    :param initial:        Boolean flag for whether to load initial weights.

    :returns: Model object with weights loaded and callbacks to use when fitting the model.
    """
    filepath: str = paths.get_model_filepath(seed, pruning_step, masks, initial)
    model: tf.keras.Model = tf.keras.models.load_model(filepath)
    return model

def save_model(model: tf.keras.Model, seed: int, pruning_step: int, masks: bool = False, initial: bool = False):
    """
    Function to save a single trained model.

    :param model:        Model object being saved.
    :param seed:         Random seed used in the model
    :param pruning_step: Integer value for the number of pruning steps which had been completed for the model.
    :param masks:        Boolean for whether the model is a real model or only masks.
    :param initial:      Boolean flag for whether this is the initial randomly initialized weights.
    """
    directory: str = paths.get_model_directory(seed, pruning_step, masks)
    paths.create_path(directory)
    filepath: str = paths.get_model_filepath(seed, pruning_step, masks, initial)
        
    # Save the initial weights in an 'initial' directory in the top-level of the model directory
    model.save(filepath, overwrite=True)

def create_lenet_300_100(
    input_shape: tuple[int, ...], 
    num_classes: int, 
    ) -> keras.Model:
    """
    Function for creating LeNet-300-100 model.

    :param input_shape: Expected input shape for images.
    :param num_classes: Number of potential classes to predict.
    :param optimizer:   Optimizer to use for training.

    :returns: Compiled LeNet-300-100 architecture.
    """
    model: keras.Model = Sequential([
        Input(input_shape),
        Dense(300, activation='relu', kernel_initializer=tf.initializers.GlorotUniform()),
        Dense(100, activation='relu', kernel_initializer=tf.initializers.GlorotUniform()),
        Dense(num_classes, activation='softmax', kernel_initializer=tf.initializers.GlorotUniform()),
    ], name="LeNet-300-100")
    
    # Explicitly build the model to initialize weights
    model.build(input_shape=input_shape)
    return model

def initialize_mask_model(model: keras.Model):
    """
    Function which performs layerwise initialization of a keras model's weights to all 1s
    as an initialization step for masking.

    :param model: Keras model being set to all 1s.
    """
        
    for weights in model.trainable_weights:
        weights.assign(
            tf.ones_like(
                input=weights,
                dtype=tf.float32,
            )
        )

def create_masked_nn(create_nn: callable, *args) -> tf.keras.Model:
    """
    Create a masked neural network where all the weights are initialized to 1s.

    :param create_nn: Function which creates a neural network model.
    :param args:      Arguments to be passed into the create_nn function.

    :returns: Stripped model with masks initialized to all 1s.
    """
    model: tf.keras.Model = create_nn(*args)
    initialize_mask_model(model)
    return model
