"""
dataset.py

File containing function(s)/classes for loading the dataset.
Note: There is a bug where this being 
"""

import functools
import os
from enum import Enum

import numpy as np
import tensorflow as tf

from src.harness import constants as C
from src.harness import utils


class Datasets(Enum):
    # MNIST images are grayscale, 28x28 pixels
    MNIST: tuple[int, int, int] = (28, 28, 1)
    # CIFAR10 images are color (RGB), 32x32 pixels
    CIFAR10: tuple[int, int, int] = (32, 32, 3)
    # ImageNet images are color (RGB), typically 224x224 pixels
    ImageNet: tuple[int, int, int] = (224, 224, 3)


class Dataset:

    def __init__(self, dataset: str | Datasets, flatten: bool = False):
        # Handle if a string is passed in, from cmdline arguments for instance
        if isinstance(dataset, str):
            match dataset.lower():
                case 'mnist':
                    self.dataset = Datasets.MNIST
                case 'cifar':
                    self.dataset = Datasets.CIFAR10
                case 'imagenet':
                    self.dataset = Datasets.ImageNet

        self.flatten = flatten
        match self.dataset:
            case Datasets.MNIST:
                self.loader_function = functools.partial(
                    load_and_process_mnist, flatten=flatten)
            case Datasets.CIFAR10:
                self.loader_function = functools.partial(
                    load_and_process_cifar10, flatten=flatten)
            case _:
                raise ValueError(
                    f'Dataset {self.dataset} is not yet supported')

    @property
    def input_shape(self):
        """
        Method to get the input shape of a dataset.

        Returns:
            tuple[int]: Shape of the dataset in length of each dimension.
        """
        shape = self.dataset.value
        if self.flatten:
            return (1, np.prod(shape))  # Flatten only the first two dimensions
        else:
            return shape

    @property
    def num_classes(self):
        """
        Method to get the number of classes in a dataset.

        Returns:
            int: Number of target classes in the dataset.
        """
        match self.dataset:
            case Datasets.MNIST:
                return 10
            case Datasets.CIFAR10:
                return 10
            case Datasets.ImageNet:
                return 1000

    @property
    def loader(self):
        """
        Method to get the data loader.

        Returns:
            callable: Function which returns the inputs/labels for the dataset
                split into training and testing sets.
        """
        return self.loader_function

    def load(self):
        """
        Method to load the data for a given dataset.

        Returns:
            Numpy data array extracted from the loader function.
        """
        return self.loader_function()


# ---------------------- Data Loader Functions ----------------------

def load_and_process_mnist(flatten: bool = False) -> tuple[np.array, np.array, np.array, np.array]:
    """
    Function to load and preprocess the MNIST dataset.
    Source: https://colab.research.google.com/github/maticvl/dataHacker/blob/master/CNN/LeNet_5_TensorFlow_2_0_datahacker.ipynb#scrollTo=UA2ehjxgF7bY

    Args:
        flatten (bool, optional): Boolean flag for whether data should be flattened to 1 dimension.
            False by default.

    Returns:
        tuple[np.array, np.array, np.array, np.array]: X and Y training and test sets after preprocessing.
    """
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

    if flatten:
        # Add a new axis for use in training the model
        X_train: np.array = X_train[:, :, :, np.newaxis]
        X_test: np.array = X_test[:, :, :, np.newaxis]

        # Reshape labels
        X_train = X_train.reshape((X_train.shape[0], 1, -1))
        X_test = X_test.reshape((X_test.shape[0], 1, -1))
        Y_train = Y_train.reshape((Y_train.shape[0], 1, -1))
        Y_test = Y_test.reshape((Y_test.shape[0], 1, -1))

    # Convert class vectors to binary class matrices.
    num_classes: int = 10
    Y_train: np.array = tf.keras.utils.to_categorical(Y_train, num_classes)
    Y_test: np.array = tf.keras.utils.to_categorical(Y_test, num_classes)

    # Data normalization
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    return X_train, X_test, Y_train, Y_test


def load_and_process_cifar10(flatten: bool = False) -> tuple[np.array, np.array, np.array, np.array]:
    """
    Function to load and preprocess the CIFAR10 dataset.

    Args:
        flatten (bool, optional): Boolean flag for whether data should be flattened to 1 dimension.
            False by default.

    Returns:
        tuple[np.array, np.array, np.array, np.array]: X and Y training and test sets after preprocessing.
    """
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()

    if flatten:
        # Add a new axis for use in training the model
        X_train: np.array = X_train[:, :, :, np.newaxis]
        X_test: np.array = X_test[:, :, :, np.newaxis]

        # Reshape labels
        X_train = X_train.reshape((X_train.shape[0], 1, -1))
        X_test = X_test.reshape((X_test.shape[0], 1, -1))
        Y_train = Y_train.reshape((Y_train.shape[0], 1, -1))
        Y_test = Y_test.reshape((Y_test.shape[0], 1, -1))

    # Convert class vectors to binary class matrices.
    num_classes: int = 10
    Y_train: np.array = tf.keras.utils.to_categorical(Y_train, num_classes)
    Y_test: np.array = tf.keras.utils.to_categorical(Y_test, num_classes)

    # Data normalization
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    return X_train, X_test, Y_train, Y_test

