"""
model.py

"""

import numpy as np
import os
import random
import tensorflow as tf
from tensorflow_model_optimization.sparsity import keras as sparsity

from src.harness.constants import Constants as C
from src.harness import paths

def load_model(seed: int, pruning_step: int, masks: bool = False) -> tf.keras.Model:
    """
    Function used to load a single trained model.

    :param seed:           Random seed the model was trained using
    :param pruning_step:   Integer value for the number of pruning steps which had been completed for the model.
    :param masks:          Boolean for whether the model masks are being retrieved or not.

    :returns: Model object with weights loaded and callbacks to use when fitting the model.
    """
    filepath: str = paths.get_model_filepath(seed, pruning_step, masks)
    model: tf.keras.Model = tf.keras.models.load_model(filepath)
    return model

def save_model(model: tf.keras.Model, seed: int, pruning_step: int, masks: bool = False):
    """
    Function to save a single trained model.

    :param model:        Model object being saved.
    :param seed:         Random seed used in the model
    :param pruning_step: Integer value for the number of pruning steps which had been completed for the model.
    :param masks:        Boolean for whether the model is a real model or only masks.
    """

    target_directory: str = paths.get_model_path(seed, pruning_step, masks)
    paths.create_path(target_directory)
    # Save the initial weights in an 'initial' directory in the top-level of the model directory
    model.save(os.path.join(target_directory, 'model.keras'), overwrite=True)

# Create a model with the same architecture using all Keras components to check its accuracy with the same parameters
def create_lenet_300_100(random_seed: int) -> tf.keras.Model:
    """
    Simple hardcoded class definition for creating the sequential Keras equivalent to LeNet-300-100.
    """
    input_shape: tuple[int, ...] = (784, )
    num_classes: int = 10

    # Set seeds for reproducability
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
# , kernel_initializer=tf.initializers.GlorotUniform()
    model = tf.keras.Sequential(name="LeNet-300-100")
    model.add(tf.keras.layers.Flatten(input_shape=input_shape))
    model.add(tf.keras.layers.Dense(300, activation='relu'))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    return model

def pruned_nn(
        random_seed: int, 
        create_model: callable,
        pruning_params: dict, 
        loss: tf.keras.losses.Loss = tf.keras.losses.categorical_crossentropy, 
        optimizer=C.OPTIMIZER()) -> tf.keras.Model:
    """
    Function to define the architecture of a neural network model
    following 300 100 architecture for MNIST dataset and using
    provided parameter which are used to prune the model.
    
    Input: 'pruning_params' Python 3 dictionary containing parameters which are used for pruning
    Output: Returns designed and compiled neural network model
    """
    # Set seeds for reproducability
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    model = sparsity.prune_low_magnitude(create_model(random_seed), **pruning_params)
    
    # Compile pruned CNN-
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model

def create_masked_nn(*args) -> tf.keras.Model:
    """
    Create a masked neural network where all the weights are initialized to 1s.
    """
    model: tf.keras.Model = pruned_nn(*args)
    model_stripped = sparsity.strip_pruning(model)
    # Assign all weights to 1 to start
    for weights in model_stripped.trainable_weights:
        weights.assign(
            tf.ones_like(
                input = weights,
                dtype = tf.float32
            )
        )
    return model
