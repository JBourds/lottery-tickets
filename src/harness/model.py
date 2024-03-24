"""
model.py

Class definition for the LeNet network architecture.
Source: https://colab.research.google.com/github/maticvl/dataHacker/blob/master/CNN/LeNet_5_TensorFlow_2_0_datahacker.ipynb#scrollTo=UA2ehjxgF7bY
"""

import functools
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D

from src.harness.constants import Constants as C
from src.harness.dataset import load_and_process_mnist
from src.lottery_ticket.foundations.model_fc import ModelFc
from src.lottery_ticket.foundations import paths
from src.lottery_ticket.foundations.save_restore import restore_network, save_network
from src.lottery_ticket.foundations import trainer

class LeNet300(ModelFc):
    def __init__(self, random_seed: int, input_placeholder: np.array, label_placeholder: np.array, presets=None, masks=None):
        self.seed: int = random_seed
        # Define the hyperparameters for LeNet-300
        hyperparameters = {
            'layers': [
                (300, tf.nn.relu),  # Fully Connected Layer with 300 units and ReLU activation
                (100, tf.nn.relu),  # Fully Connected Layer with 100 units and ReLU activation
                (10, None)          # Output Layer with 10 units (for 10 classes) and no activation
            ]
        }
        
        # Call parent constructor with LeNet-300 hyperparameters
        super(LeNet300, self).__init__(hyperparameters=hyperparameters,
                                        input_placeholder=input_placeholder,
                                        label_placeholder=label_placeholder,
                                        presets=presets,
                                        masks=masks)

def load_model(model_index: int, pruning_step: int, untrained: bool = False) -> LeNet300:
    """
    Function used to load a single trained model.

    :param model_index:    Index of the model which was trained.
    :param pruning_step:   Integer value for the number of pruning steps which had been completed for the model.
    :param untrained:      Boolean to get the trained or untrained version of a model.

    :returns: Model object with weights loaded and callbacks to use when fitting the model.
    """
    path: str = paths.get_model_directory(model_index, C.MODEL_DIRECTORY)
    if untrained:
        path = paths.initial(path)
    else:
        path = paths.trial(path, pruning_step)
    X_train, Y_train, _, _ = load_and_process_mnist()
    model: LeNet300 = LeNet300(model_index, X_train, Y_train)
    model._weights = restore_network(paths.weights(path))
    model._masks = restore_network(paths.masks(path))
    return model

def save_model(model: LeNet300, pruning_step: int, untrained: bool = False, final: bool = False):
    """
    Function to save a single trained model.

    :param model:        Model object being saved.
    :param pruning_step: Integer value for the number of pruning steps which had been completed for the model.
    :param untrained:    Boolean for if it is the untrained version of a model. Defaults to False.
    :param final:        Boolean for if it is the final version of a model. Defaults to False.
    """

    output_directory: str = paths.get_model_directory(model.seed, C.MODEL_DIRECTORY)
    # Save the initial weights in an 'initial' directory in the top-level of the model directory
    if untrained:
        untrained_directory: str = paths.initial(output_directory)
        initial_masks: dict[str: np.array] = {key: np.ones(layer.shape) for key, layer in model._weights.items()}
        save_network(paths.masks(untrained_directory), initial_masks)
        save_network(paths.weights(untrained_directory), model.weights)
    elif final:
        final_directory: str = paths.final(output_directory)
        save_network(paths.masks(final_directory), model.masks)
        save_network(paths.weights(final_directory), model.weights)
    else:
        # Create a trial directory within the model directory
        trial_directory: str = paths.trial(output_directory, pruning_step)
        # Save model and weights to the trial directory
        save_network(paths.weights(trial_directory), model.weights)
        save_network(paths.masks(trial_directory), model.masks)

def create_model(random_seed: int, X_train: np.array, Y_train: np.array) -> LeNet300:
    """
    Method used for setting the random seed(s) and instantiating a model then saving its pretrained weights.

    :param random_seed:  Value used to ensure reproducability.
    :param X_train:      Numpy array for placeholder input.
    :param Y_train:      Numpy array for placeholder labels.  

    :returns: Model and callbacks.
    """
    # Set seeds
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    # Initialize the model
    model: LeNet300 = LeNet300(random_seed, X_train, Y_train)
    
    return model
