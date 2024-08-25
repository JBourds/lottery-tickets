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
from sys import platform
from typing import Callable, Tuple

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
    optimizer: Callable[[None], tf.keras.optimizers.Optimizer]
    loss_function: Callable[[None], tf.keras.losses.Loss]
    accuracy_metric: Callable[[None], tf.keras.metrics.Metric]
    epochs: int
    batch_size: int
    fc_pruning_rate: float
    conv_pruning_rate: float
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
    @staticmethod
    def lenet_300_100_hyperparameters() -> Hyperparameters:
        return Hyperparameters(
            patience=2,
            minimum_delta=0.0001,
            learning_rate=1.2e-3,
            optimizer=Adam,
            loss_function=tf.keras.losses.CategoricalCrossentropy,
            accuracy_metric=tf.keras.metrics.CategoricalAccuracy,
            epochs=60,
            batch_size=60,
            fc_pruning_rate=0.2,
            conv_pruning_rate=0,
            eval_freq=100,
            early_stopping=True,
        )

    @staticmethod
    def conv2_hyperparameters() -> Hyperparameters:
        return Hyperparameters(
            patience=2,
            minimum_delta=0.0001,
            learning_rate=2e-4,
            optimizer=Adam,
            loss_function=tf.keras.losses.CategoricalCrossentropy,
            accuracy_metric=tf.keras.metrics.CategoricalAccuracy,
            epochs=20_000,
            batch_size=60,
            fc_pruning_rate=0.2,
            conv_pruning_rate=0.1,
            eval_freq=100,
            early_stopping=True,
        )

    @staticmethod
    def conv4_hyperparameters() -> Hyperparameters:
        return Hyperparameters(
            patience=2,
            minimum_delta=0.0001,
            learning_rate=3e-4,
            optimizer=Adam,
            loss_function=tf.keras.losses.CategoricalCrossentropy,
            accuracy_metric=tf.keras.metrics.CategoricalAccuracy,
            epochs=25_000,
            batch_size=60,
            fc_pruning_rate=0.2,
            conv_pruning_rate=0.1,
            eval_freq=100,
            early_stopping=True,
        )

    @staticmethod
    def conv6_hyperparameters() -> Hyperparameters:
        return Hyperparameters(
            patience=2,
            minimum_delta=0.0001,
            learning_rate=3e-4,
            optimizer=Adam,
            loss_function=tf.keras.losses.CategoricalCrossentropy,
            accuracy_metric=tf.keras.metrics.CategoricalAccuracy,
            iterations=30_000,
            batch_size=60,
            fc_pruning_rate=0.2,
            conv_pruning_rate=0.15,
            eval_freq=100,
            early_stopping=True,
        )

  # --------------- Constructors ---------------

    @staticmethod
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
            Dense(num_classes, activation='softmax',
                  kernel_initializer=initializer),
        ], name="LeNet-300-100")

        # Explicitly build the model to initialize weights
        if platform == 'darwin':
            model.build(input_shape=input_shape)
        return model

    @staticmethod
    def create_conv2(
            input_shape: Tuple[int, ...],
            num_classes: int,
            initializer: tf.initializers.Initializer = tf.initializers.GlorotNormal(),
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
        ], name="Conv-2")

        # Explicitly build the model to initialize weights
        if platform == 'darwin':
            model.build(input_shape=input_shape)
        return model

    @staticmethod
    def create_conv4(
            input_shape: Tuple[int, ...],
            num_classes: int,
            initializer: tf.initializers.Initializer = tf.initializers.GlorotNormal(),
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

        ], name="Conv-4")

        # Explicitly build the model to initialize weights
        if platform == 'darwin':
            model.build(input_shape=input_shape)
        return model

    @staticmethod
    def create_conv6(
            input_shape: Tuple[int, ...],
            num_classes: int,
            initializer: tf.initializers.Initializer = tf.initializers.GlorotNormal(),
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
        ], name="Conv-6")

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

    def __init__(self, architecture: str, dataset: ds.Datasets):
        self.architecture = Architectures(architecture.lower())
        if not self._supported(self.architecture):
            raise ValueError(
                f"'{self.architecture}' is not a supported architecture")
        self.dataset = ds.Dataset(dataset, flatten=not self.convolutional())

    def convolutional(self) -> bool:
        return self._convolutional(self.architecture)

    @staticmethod
    def _supported(architecture: Architectures) -> bool:
        return architecture in set(Architecture.CONSTRUCTORS.keys()).intersection(Architecture.HYPERPARAMETERS.keys())

    @staticmethod
    def _convolutional(architecture: Architectures) -> bool:
        # Lazy implementation for now
        return architecture != Architectures.LENET

    def get_model_hyperparameters(self) -> Hyperparameters:
        hyperparameters = self.HYPERPARAMETERS.get(self.architecture)
        if hyperparameters is None:
            raise NotImplementedError(
                f"'{self.architecture}' not implemented in Architecture class")
        return hyperparameters()

    def get_model_constructor(self) -> Callable[[None], keras.models.Model]:
        constructor_function = self.CONSTRUCTORS.get(self.architecture)
        if constructor_function is None:
            raise NotImplementedError(
                f"'{self.architecture}' not implemented in Architecture class")
        return functools.partial(constructor_function, self.dataset.input_shape, self.dataset.num_classes)
