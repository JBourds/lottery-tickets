"""
dataset.py

File containing function(s)/classes for loading the dataset.
"""

import numpy as np
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical

def print_dataset_shape(X_train: np.array, Y_train: np.array, X_test: np.array, Y_test: np.array):
    """
    Function to print the shape of the dataset.

    :param X_train: Training data.
    :param Y_train: Training labels.
    :param X_test:  Testing data.
    :param Y_test:  Testing labels.
    """
    print('x_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    print(X_train[0].shape, 'image shape')

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    print(X_train[0].shape, 'image shape')

def load_and_process_mnist() -> tuple[np.array, np.array, np.array, np.array]:
    """
    Function to load and preprocess the MNIST dataset.
    Source: https://colab.research.google.com/github/maticvl/dataHacker/blob/master/CNN/LeNet_5_TensorFlow_2_0_datahacker.ipynb#scrollTo=UA2ehjxgF7bY

    :returns X and Y training and test sets after preprocessing.
    """
    (X_train, Y_train), (X_test, Y_test) = datasets.mnist.load_data()
    
    # Add a new axis for use in training the model
    X_train: np.array = X_train[:, :, :, np.newaxis]
    X_test: np.array = X_test[:, :, :, np.newaxis]

    # Convert class vectors to binary class matrices.
    num_classes: int = 10
    Y_train: np.array = to_categorical(Y_train, num_classes)
    Y_test: np.array = to_categorical(Y_test, num_classes)

    # Data normalization
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    return X_train, Y_train, X_test, Y_test