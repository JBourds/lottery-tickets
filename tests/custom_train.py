"""
Testing custom training loop to verify it works.
"""

import logging
import sys

import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow import keras

sys.path.append('../')

if __name__ == '__main__':
    from src.harness import model as mod
    from src.harness import training
    from src.harness.architecture import Architecture, Hyperparameters

    logging.getLogger().setLevel(logging.DEBUG)

    architecture = Architecture('conv2', 'cifar')
    model = architecture.get_model_constructor()()
    mask_model = mod.create_masked_nn(architecture.get_model_constructor())
    dataset = architecture.dataset
    hyperparams = architecture.get_model_hyperparameters()
    X_train, X_test, Y_train, Y_test = dataset.load()

    trial_data = training.train(0, 0, model, mask_model, dataset, hyperparams)
