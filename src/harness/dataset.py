# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
dataset.py

File containing function(s)/classes for loading the dataset.
"""

import numpy as np
import os
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical

from src.harness.constants import Constants as C

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

def download_data(dataset_directory: str = C.MNIST_LOCATION):
    """
    Function to download the MNIST dataset.

    :param dataset_directory: Name of the dataset's folder.
    """
    X_train, Y_train, X_test, Y_test = load_and_process_mnist()
    os.makedirs(dataset_directory, exist_ok=True)
    np.save(f'{dataset_directory}x_train.npy', X_train)
    np.save(f'{dataset_directory}y_train.npy', Y_train)
    np.save(f'{dataset_directory}x_test.npy', X_test)
    np.save(f'{dataset_directory}y_test.npy', Y_test)



def load_and_process_mnist(flatten: bool = True) -> tuple[np.array, np.array, np.array, np.array]:
    """
    Function to load and preprocess the MNIST dataset.
    Source: https://colab.research.google.com/github/maticvl/dataHacker/blob/master/CNN/LeNet_5_TensorFlow_2_0_datahacker.ipynb#scrollTo=UA2ehjxgF7bY

    :param flatten: Boolean value for if the input data should be flattened.

    :returns X and Y training and test sets after preprocessing.
    """
    (X_train, Y_train), (X_test, Y_test) = datasets.mnist.load_data()
    
    # Add a new axis for use in training the model
    X_train: np.array = X_train[:, :, :, np.newaxis]
    X_test: np.array = X_test[:, :, :, np.newaxis]

    # Flatten labels
    if flatten:
      X_train = X_train.reshape((X_train.shape[0], -1))
      X_test = X_test.reshape((X_test.shape[0], -1))

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