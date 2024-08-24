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
from keras.layers import Dense, Flatten, Input
from sys import platform
from typing import Callable

from src.harness import constants as C
from src.harness import dataset as ds

def create_lenet_300_100(
    input_shape: tuple[int, ...], 
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
    model: keras.Model = Sequential([
        Input(input_shape),
        Dense(300, activation='relu', kernel_initializer=initializer),
        Dense(100, activation='relu', kernel_initializer=initializer),
        Dense(num_classes, activation='softmax', kernel_initializer=initializer),
    ], name="LeNet-300-100")
    
    # Explicitly build the model to initialize weights
    if platform == 'darwin':
        model.build(input_shape=input_shape)
    return model

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
                constructor_function = None
            case 'conv-4':
                constructor_function = None
            case 'conv-6':
                constructor_function = None
            case _:
                raise NotImplementedError(f"'{self.architecture}' not implemented in Architecture class")
        return functools.partial(constructor_function, self.dataset.input_shape, self.dataset.num_classes)
