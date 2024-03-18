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

import src.harness.constants as C
from src.harness.dataset import load_and_process_mnist
from src.lottery_ticket.foundations.model_fc import ModelFc
from src.lottery_ticket.foundations.paths import create_path, get_model_directory, initial, trial
from src.lottery_ticket.foundations.save_restore import restore_network, save_network
from src.lottery_ticket.foundations import trainer

class LeNet300(ModelFc):
    def __init__(self, input_placeholder, label_placeholder, presets=None, masks=None):
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
    path: str = get_model_directory(model_index, C.MODEL_DIRECTORY)
    if untrained:
        path = initial(path)
    path = trial(path, pruning_step)
    X_train, Y_train, _, _ = load_and_process_mnist()
    model: LeNet300 = LeNet300(X_train, Y_train)
    model._weights = restore_network(path)
    return model

def save_model(model: LeNet300, model_index: int, pruning_step: int, untrained: bool = False):
    """
    Function to save a single trained model.

    :param model:        Model object being saved.
    :param model_index:  Index of the model which was trained.
    :param pruning_step: Integer value for the number of pruning steps which had been completed for the model.
    :param untrained:    Boolean for if it is the untrained version of a model.
    """

    output_directory: str = get_model_directory(model_index, C.MODEL_DIRECTORY)
    if untrained:
       output_directory = initial(output_directory)
    create_path(output_directory)
    model_name: str = trial(output_directory, pruning_step)
    save_network(model_name, model.weights)

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
    model: LeNet300 = LeNet300(X_train, Y_train)

    # Save the untrained weights
    save_model(model, random_seed, 0, True)
    
    return model

def train(model_index: int,
          output_dir: str,
          training_len: int = C.TRAINING_LENGTH,
          pruning_steps: int = C.TRAINING_ITERATIONS,
          presets=None,
          permute_labels: bool = False):
  """
  Perform the lottery ticket experiment.

  The output of each experiment will be stored in a directory called:
  {models_dir}/{model_num}/{experiment_name} as defined in the
  foundations.paths module.

    :param model_index: Integer value for the model index (random seed).
    :param training_len: How long to train on each iteration.
    :param pruning_steps: How many iterative pruning steps to perform.
    :param presets: The initial weights for the network, if any. 
                    Presets can come in one of three forms:
        * A dictionary of numpy arrays. Each dictionary key is the name of the
            corresponding tensor that is to be initialized. Each value is a numpy
            array containing the initializations.
        * The string name of a directory containing one file for each
            set of weights that is to be initialized (in the form of
            foundations.save_restore).
        * None, meaning the network should be randomly initialized.
    :param permute_labels: Whether to permute the labels on the dataset.
  """
  # Define model and dataset functions.
  make_dataset = load_and_process_mnist
  make_model = functools.partial(create_model, 0)

  
