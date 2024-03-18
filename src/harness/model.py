"""
model.py

Class definition for the LeNet network architecture.
Source: https://colab.research.google.com/github/maticvl/dataHacker/blob/master/CNN/LeNet_5_TensorFlow_2_0_datahacker.ipynb#scrollTo=UA2ehjxgF7bY
"""

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
from src.harness.utils import create_path, get_model_directory, get_model_name
from src.lottery_ticket.foundations.model_fc import ModelFc
from src.lottery_ticket.foundations.save_restore import restore_network, save_network

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

def load_model(model_index: int, pruning_step: int, trained) -> LeNet300:
    """
    Function used to load a single trained model.

    :param model_index:       Index of the model which was trained.
    :param pruning_step:      Integer value for the number of pruning steps which had been completed for the model.
    :param trained:           Boolean to get the trained or untrained version of a model.

    :returns: Model object with weights loaded and callbacks to use when fitting the model.
    """
    path: str = get_model_directory(model_index, C.MODEL_DIRECTORY) + get_model_name(model_index, pruning_step, trained)
    model: LeNet300 = LeNet300()
    model.weights = restore_network(path)
    return model

def save_model(model: LeNet300, model_index: int, pruning_step: int, trained: bool):
    """
    Function to save a single trained model.

    :param model:        Model object being saved.
    :param model_index:  Index of the model which was trained.
    :param pruning_step: Integer value for the number of pruning steps which had been completed for the model.
    :param trained:      Boolean to get the trained or untrained version of a model.
    """

    output_directory: str = get_model_directory(model_index, C.MODEL_DIRECTORY)
    create_path(output_directory)
    model_name: str = get_model_name(model_index, pruning_step, trained)
    save_network(output_directory + model_name, model.weights)

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

    # Save the pretrained weights
    save_model(model, random_seed, 0, False)
    
    return model

def create_models(X_train: np.array, Y_train: np.array, X_test: np.array, Y_test: np.array, epochs: int, num_models: int):
    """
    Function responsible for training/saving the base, fully parametrized models.

    :param X_train:     Training instances.
    :param X_test:      Testing instances.
    :param Y_train:     Training labels.
    :param Y_test:      Testing labels.
    :param epochs:      Number of epochs to train the model for.
    :param num_models:  Number of models to create.
    """
    assert X_train.shape[0] >= 1, 'Need at least one input to determine feature shape'

    # Extract shape of features and the number of classes
    feature_shape: tuple[int, ...] = X_train[0].shape
    num_classes: int = 10

    # Use index as the random seed input
    for i in range(num_models):
        # Create the model if it does not already exist
        if not os.path.exists(get_model_directory(i, C.MODEL_DIRECTORY) + get_model_name(i, 0)):
            # Setup and train the model
            model, callbacks = create_model(feature_shape, num_classes, random_seed=i)
            model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_test, Y_test), callbacks=callbacks, verbose=1, use_multiprocessing=True) 
            save_model(model, i, 0, True)