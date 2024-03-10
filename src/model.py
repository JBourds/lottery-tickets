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
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D

import src.constants as C
from src.utils import create_path, get_model_callbacks, get_model_directory, get_model_name

class LeNet(Sequential):

    def __init__(self, input_shape: np.array, num_classes: int, **kwargs):
        """
        LeNet-5 model derived from this paper: http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf

        :param input_shape: Shape of the input instances.
        :param num_classes: Number of classes in the dataset.
        """
        super().__init__()

        # Convolutional layers  
        self.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=input_shape, padding="same"))
        self.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
        self.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.add(Flatten())

        # Fully connected output layers
        self.add(Dense(120, activation='tanh'))
        self.add(Dense(84, activation='tanh'))
        self.add(Dense(num_classes, activation='softmax'))

        self.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def load_model(feature_shape: tuple[int, ...], num_classes: int, model_index: int, pruning_step: int, trained) -> tuple[LeNet, list[tf.keras.callbacks]]:
    """
    Function used to load a single trained model.

    :param feature_shape:     Tuple of integer dimensions for the feature shape.
    :param num_classes:       Number of unique classes for the model.
    :param model_index:       Index of the model which was trained.
    :param pruning_step:      Integer value for the number of pruning steps which had been completed for the model.
    :param trained:           Boolean to get the trained or untrained version of a model.

    :returns: Model object with weights loaded and callbacks to use when fitting the model.
    """
    path: str = get_model_directory(model_index, C.MODEL_DIRECTORY) + get_model_name(model_index, pruning_step, trained)
    model: LeNet = LeNet(feature_shape, num_classes)
    model.load_weights(path)
    callbacks: list[tf.keras.callbacks] = get_model_callbacks(model_index, pruning_step)   

    return model, callbacks

def save_model(model: LeNet, model_index: int, pruning_step: int, trained: bool):
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
    model.save(output_directory + model_name)

def create_model(feature_shape: tuple[int, ...], num_classes: int, random_seed: int) -> tuple[LeNet, list[tf.keras.callbacks]]:
    """
    Method used for setting the random seed(s) and instantiating a model then saving its pretrained weights.

    :param feature_shape:  Shape of the features.
    :param num_classes:    Number of potential classes. 10 for MNIST.
    :param random_seed:    Value used to ensure reproducability.

    :returns: Model and callbacks.
    """
    # Set seeds
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    # Initialize the model
    model: LeNet = LeNet(feature_shape, num_classes)
    callbacks: list[tf.keras.callbacks] = get_model_callbacks(random_seed, 0)

    # Save the pretrained weights
    save_model(model, random_seed, 0, False)
    
    return model, callbacks

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