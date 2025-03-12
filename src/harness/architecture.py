"""
architecture.py

File containing the architecture class which provides the interface for
interacting with a specific architecture along with all the creation functions
for supported architectures.

Author: Jordan Bourdeau
"""

import functools
from dataclasses import dataclass
from enum import Enum
import numpy as np
from sys import platform
from typing import Callable, List, Tuple

import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow import keras

from src.harness import constants as C
from src.harness import dataset as ds

Adam = tf.keras.optimizers.legacy.Adam if platform == 'darwin' else tf.keras.optimizers.Adam


@dataclass
class Hyperparameters:
    """
    Class representing hyperparameters which get fed into training data.
    """
    patience: int
    minimum_delta: float
    learning_rate: float
    optimizer: Callable[[], tf.keras.optimizers.Optimizer]
    loss_function: Callable[[], tf.keras.losses.Loss]
    accuracy_metric: Callable[[], tf.keras.metrics.Metric]
    epochs: int
    batch_size: int
    # The number of iterations (batches) to train for at a time before
    # evaluating validation set performance and checking for early stopping
    eval_freq: int
    early_stopping: bool


class Architectures(Enum):
    """
    Class representing the supported architectures, along with the functions
    to use in building and training them.
    """
    LENET = 'lenet'
    CONV_2 = 'conv2'
    CONV_4 = 'conv4'
    CONV_6 = 'conv6'

    # --------------- Hyperparameters ---------------
    def lenet_300_100_hyperparameters(**kwargs) -> Hyperparameters:
        return Hyperparameters(
            patience=kwargs.get('patience', 5),
            minimum_delta=kwargs.get('minimum_delta', 0.001),
            learning_rate=kwargs.get('learning_rate', 1.2e-3),
            optimizer=kwargs.get('optimizer', Adam),
            loss_function=kwargs.get(
                'loss', tf.keras.losses.CategoricalCrossentropy),
            accuracy_metric=kwargs.get(
                'accuracy', tf.keras.metrics.CategoricalAccuracy),
            epochs=kwargs.get('epochs', 60),
            batch_size=kwargs.get('batch_size', 60),
            eval_freq=kwargs.get('eval_freq', 100),
            early_stopping=kwargs.get('early_stopping', False),
        )

    @staticmethod
    def conv2_hyperparameters(**kwargs) -> Hyperparameters:
        return Hyperparameters(
            patience=kwargs.get('patience', 5),
            minimum_delta=kwargs.get('minimum_delta', 0.001),
            learning_rate=kwargs.get('learning_rate', 2e-4),
            optimizer=kwargs.get('optimizer', Adam),
            loss_function=kwargs.get(
                'loss', tf.keras.losses.CategoricalCrossentropy),
            accuracy_metric=kwargs.get(
                'accuracy', tf.keras.metrics.CategoricalAccuracy),
            epochs=kwargs.get('epochs', 60),
            batch_size=kwargs.get('batch_size', 60),
            eval_freq=kwargs.get('eval_freq', 100),
            early_stopping=kwargs.get('early_stopping', False),
        )

    @staticmethod
    def conv4_hyperparameters(**kwargs) -> Hyperparameters:
        return Hyperparameters(
            patience=kwargs.get('patience', 5),
            minimum_delta=kwargs.get('minimum_delta', 0.001),
            learning_rate=kwargs.get('learning_rate', 3e-4),
            optimizer=kwargs.get('optimizer', Adam),
            loss_function=kwargs.get(
                'loss', tf.keras.losses.CategoricalCrossentropy),
            accuracy_metric=kwargs.get(
                'accuracy', tf.keras.metrics.CategoricalAccuracy),
            epochs=kwargs.get('epochs', 60),
            batch_size=kwargs.get('batch_size', 60),
            eval_freq=kwargs.get('eval_freq', 100),
            early_stopping=kwargs.get('early_stopping', False),
        )

    @staticmethod
    def conv6_hyperparameters(**kwargs) -> Hyperparameters:
        return Hyperparameters(
            patience=kwargs.get('patience', 5),
            minimum_delta=kwargs.get('minimum_delta', 0.001),
            learning_rate=kwargs.get('learning_rate', 3e-4),
            optimizer=kwargs.get('optimizer', Adam),
            loss_function=kwargs.get(
                'loss', tf.keras.losses.CategoricalCrossentropy),
            accuracy_metric=kwargs.get(
                'accuracy', tf.keras.metrics.CategoricalAccuracy),
            epochs=kwargs.get('epochs', 60),
            batch_size=kwargs.get('batch_size', 60),
            eval_freq=kwargs.get('eval_freq', 100),
            early_stopping=kwargs.get('early_stopping', False),
        )

  # --------------- Constructors ---------------

    @staticmethod
    def create_lenet_300_100(
        input_shape: Tuple[int, ...],
        num_classes: int,
    ) -> keras.Model:
        """
        Function for creating LeNet-300-100 model.

        :param input_shape: Expected input shape for images.
        :param num_classes: Number of potential classes to predict.

        :returns: Compiled LeNet-300-100 architecture.
        """
        model = Sequential([
            Input(input_shape),
            Dense(300, activation='relu'),
            Dense(100, activation='relu'),
            Dense(num_classes, activation='softmax'),
        ], name="lenet")

        # Explicitly build the model to initialize weights
        if platform == 'darwin':
            model.build(input_shape=input_shape)
        return model

    @staticmethod
    def create_conv2(
        input_shape: Tuple[int, ...],
        num_classes: int,
    ) -> keras.Model:
        model = Sequential([
            Input(input_shape),
            Conv2D(filters=64, kernel_size=(3, 3),
                   padding="same", activation="relu"),
            Conv2D(filters=64, kernel_size=(3, 3),
                   padding="same", activation="relu"),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Flatten(),
            Dense(4086, activation='relu'),
            Dense(4086, activation='relu'),
            Dense(num_classes, activation='softmax'),
        ], name="conv2")

        # Explicitly build the model to initialize weights
        if platform == 'darwin':
            model.build(input_shape=input_shape)
        return model

    @staticmethod
    def create_conv4(
        input_shape: Tuple[int, ...],
        num_classes: int,
    ) -> keras.Model:
        model = Sequential([
            Input(input_shape),
            Conv2D(filters=64, kernel_size=(3, 3),
                   padding="same", activation="relu"),
            Conv2D(filters=64, kernel_size=(3, 3),
                   padding="same", activation="relu"),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(filters=128, kernel_size=(3, 3),
                   padding="same", activation="relu"),
            Conv2D(filters=128, kernel_size=(3, 3),
                   padding="same", activation="relu"),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Flatten(),
            Dense(4086, activation='relu'),
            Dense(4086, activation='relu'),
            Dense(num_classes, activation='softmax'),

        ], name="conv4")

        # Explicitly build the model to initialize weights
        if platform == 'darwin':
            model.build(input_shape=input_shape)
        return model

    @staticmethod
    def create_conv6(
        input_shape: Tuple[int, ...],
        num_classes: int,
    ) -> keras.Model:
        model = Sequential([
            Input(input_shape),
            Conv2D(filters=64, kernel_size=(3, 3),
                   padding="same", activation="relu"),
            Conv2D(filters=64, kernel_size=(3, 3),
                   padding="same", activation="relu"),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(filters=128, kernel_size=(3, 3),
                   padding="same", activation="relu"),
            Conv2D(filters=128, kernel_size=(3, 3),
                   padding="same", activation="relu"),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(filters=256, kernel_size=(3, 3),
                   padding="same", activation="relu"),
            Conv2D(filters=256, kernel_size=(3, 3),
                   padding="same", activation="relu"),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Flatten(),
            Dense(4086, activation='relu'),
            Dense(4086, activation='relu'),
            Dense(num_classes, activation='softmax'),
        ], name="conv6")

        # Explicitly build the model to initialize weights
        if platform == 'darwin':
            model.build(input_shape=input_shape)
        return model


@dataclass
class Architecture:
    CONSTRUCTORS = {
        Architectures.LENET: Architectures.create_lenet_300_100,
        Architectures.CONV_2: Architectures.create_conv2,
        Architectures.CONV_4: Architectures.create_conv4,
        Architectures.CONV_6: Architectures.create_conv6,
    }

    HYPERPARAMETERS = {
        Architectures.LENET: Architectures.lenet_300_100_hyperparameters,
        Architectures.CONV_2: Architectures.conv2_hyperparameters,
        Architectures.CONV_4: Architectures.conv4_hyperparameters,
        Architectures.CONV_6: Architectures.conv6_hyperparameters,
    }

    LAYER_TYPES = ["dense", "bias", "conv", "output"]

    LAYERS = {
        Architectures.LENET: [
            "Dense 1 (300)", "Bias 1",
            "Dense 2 (100)", "Bias 2",
            "Output", "Bias 3",
        ],
        Architectures.CONV_2: [
            "Conv 1", "Bias 1",
            "Conv 2", "Bias 2",
            "Dense 1 (4086)", "Bias 3",
            "Dense 2 (4086)", "Bias 4",
            "Output", "Bias 5",
        ],
        Architectures.CONV_4: [
            "Conv 1", "Bias1",
            "Conv 2", "Bias2",
            "Conv 3", "Bias3",
            "Conv 4", "Bias4",
            "Dense 1 (4086)", "Bias 5",
            "Dense 2 (4086)", "Bias 6",
            "Output", "Bias 7",
        ],
        Architectures.CONV_6: [
            "Conv 1", "Bias 1",
            "Conv 2", "Bias 2",
            "Conv 3", "Bias 3",
            "Conv 4", "Bias 4",
            "Conv 5", "Bias 5",
            "Conv 6", "Bias 6",
            "Dense 1 (4086)", "Bias 7",
            "Dense 1 (4086)", "Bias 8",
            "Output", "Bias 9",
        ],
    }

    def __init__(self, architecture: str, dataset: str):
        self.architecture = Architectures(architecture.lower())
        if not self._supported(self.architecture):
            raise ValueError(
                f"'{self.architecture}' is not a supported architecture")
        self.dataset = ds.Dataset(dataset, flatten=not self.convolutional())

    @property
    def layer_names(self) -> List[str]:
        return self.LAYERS[self.architecture]

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.dataset.load()

    def convolutional(self) -> bool:
        return self._convolutional(self.architecture)

    @staticmethod
    def _supported(architecture: Architectures) -> bool:
        return architecture in set(Architecture.CONSTRUCTORS.keys()).intersection(Architecture.HYPERPARAMETERS.keys())

    @staticmethod
    def _convolutional(architecture: Architectures) -> bool:
        # Lazy implementation for now
        return architecture != Architectures.LENET

    def get_model_hyperparameters(self, **kwargs) -> Hyperparameters:
        hyperparameters = self.HYPERPARAMETERS.get(self.architecture)
        if hyperparameters is None:
            raise NotImplementedError(
                f"'{self.architecture}' not implemented in Architecture class")
        return hyperparameters(**kwargs)

    def get_model_constructor(self) -> Callable[[], keras.models.Model]:
        constructor_function = self.CONSTRUCTORS.get(self.architecture)
        if constructor_function is None:
            raise NotImplementedError(
                f"'{self.architecture}' not implemented in Architecture class")
        return functools.partial(constructor_function, self.dataset.input_shape, self.dataset.num_classes)

    @staticmethod
    def get_model_layers(architecture: str) -> List[str]:
        architecture = Architectures(architecture)
        layers = Architecture.LAYERS.get(architecture)
        if layers is None:
            raise NotImplementedError(
                f"'{architecture}' not implemented in Architecture class")
        return layers

    @staticmethod
    def ohe_layer_types(architecture: str) -> np.ndarray[np.ndarray[np.int8]]:
        model_layers = Architecture.get_model_layers(architecture)
        outputs = np.zeros((len(model_layers), len(Architecture.LAYER_TYPES)))
        for layer_index, layer_name in enumerate(model_layers):
            for column_index, column_name in enumerate(Architecture.LAYER_TYPES):
                if column_name in layer_name.lower():
                    outputs[layer_index, column_index] = 1
                    break
        return outputs
