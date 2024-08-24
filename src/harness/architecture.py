"""
architecture.py

File containing the architecture class which provides the interface for
interacting with a specific architecture along with all the creation functions
for supported architectures.

Author: Jordan Bourdeau
"""

from dataclasses import dataclass
import functools
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from sys import platform
from typing import Callable, Tuple

from src.harness import constants as C
from src.harness import dataset as ds


@dataclass
class Architecture:
    def __init__(self, architecture: str, dataset: ds.Datasets):
        self.architecture = architecture.lower()
        if self.architecture not in C.SUPPORTED_ARCHITECTURES:
            raise ValueError(f"'{self.architecture}' is not a supported architecture")
        self.dataset = ds.Dataset(dataset, flatten=not self.convolutional())

    def convolutional(self) -> bool:
        return self.is_convolutional(self.architecture)
    
    @staticmethod
    def is_convolutional(architecture: str) -> bool:
        # Lazy implementation for now
        return architecture.lower() not in C.MLP_ARCHITECTURES

    def get_model_constructor(self) -> Callable[None, keras.models.Model]:
        match self.architecture:
            case 'lenet':
                constructor_function = create_lenet_300_100
            case 'conv-2':
                constructor_function = create_conv2
            case 'conv-4':
                constructor_function = create_conv4
            case 'conv-6':
                constructor_function = create_conv6
            case _:
                raise NotImplementedError(f"'{self.architecture}' not implemented in Architecture class")
        return functools.partial(constructor_function, self.dataset.input_shape, self.dataset.num_classes)

def create_lenet_300_100(
    input_shape: Tuple[int, ...], 
    num_classes: int, 
    initializer: tf.initializers.Initializer = tf.initializers.GlorotNormal(),
    ) -> keras.Model:
    """
    Function for creating LeNet-300-100 model.

    :param input_shape: Expected input shape for images.
    :param num_classes: Number of potential classes to predict.
    :param initializer: Initializer used to set weights at the beginning

    :returns: Compiled LeNet-300-100 architecture.
    """
    model = Sequential([
        Input(input_shape),
        Dense(300, activation='relu', kernel_initializer=initializer),
        Dense(100, activation='relu', kernel_initializer=initializer),
        Dense(num_classes, activation='softmax', kernel_initializer=initializer),
    ], name="LeNet-300-100")
    
    # Explicitly build the model to initialize weights
    if platform == 'darwin':
        model.build(input_shape=input_shape)
    return model

def create_conv2(
    input_shape: Tuple[int, ...], 
    num_classes: int,
    initializer: tf.initializers.Initializer = tf.initializers.GlorotNormal(),
    ) -> keras.Model:
    model = Sequential([
        Input(input_shape),
        Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
        Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Flatten(),
        Dense(4086, activation='relu'),
        Dense(4086, activation='relu'),
        Dense(num_classes, activation='softmax'),
    ], name="Conv-2")

    # Explicitly build the model to initialize weights
    if platform == 'darwin':
        model.build(input_shape=input_shape)
    return model

def create_conv4(
    input_shape: Tuple[int, ...], 
    num_classes: int,
    initializer: tf.initializers.Initializer = tf.initializers.GlorotNormal(),
    ) -> keras.Model:
    model = Sequential([
        Input(input_shape),
        Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
        Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
        Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Flatten(),
        Dense(4086, activation='relu'),
        Dense(4086, activation='relu'),
        Dense(num_classes, activation='softmax'),

    ], name="Conv-4")

    # Explicitly build the model to initialize weights
    if platform == 'darwin':
        model.build(input_shape=input_shape)
    return model

def create_conv6(
    input_shape: Tuple[int, ...], 
    num_classes: int,
    initializer: tf.initializers.Initializer = tf.initializers.GlorotNormal(),
    ) -> keras.Model:
    model = Sequential([
        Input(input_shape),
        Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
        Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
        Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
        Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Flatten(),
        Dense(4086, activation='relu'),
        Dense(4086, activation='relu'),
        Dense(num_classes, activation='softmax'),
    ], name="Conv-6")

    # Explicitly build the model to initialize weights
    if platform == 'darwin':
        model.build(input_shape=input_shape)
    return model

def create_resnet18(
    input_shape: Tuple[int, ...], 
    num_classes: int,
    initializer: tf.initializers.Initializer = tf.initializers.GlorotNormal(),
    ) -> keras.Model:


    model = Sequential([
    ], name="Resnet-18")

    # Explicitly build the model to initialize weights
    if platform == 'darwin':
        model.build(input_shape=input_shape)
    return model


