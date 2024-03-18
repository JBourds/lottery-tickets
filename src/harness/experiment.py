"""
experiment.py

Module containing code for actually running the lottery ticket hypothesis experiemnts.
"""

import numpy as np

from src.harness.model import create_models
from src.harness.pruning import iterative_magnitude_pruning

def create_lottery_tickets(X_train: np.array, Y_train: np.array, X_test: np.array, Y_test: np.array, total_pruning_percentage: float, pruning_steps: int, epochs_per_step: int, num_models: int):
    """
    Function to perform the lottery ticket hypothesis with iterative magnitude pruning.
    Does not return anything, but saves all steps in training/pruning models

    :param X_train:                  Training instances.
    :param X_test:                   Testing instances.
    :param Y_train:                  Training labels.
    :param Y_test:                   Testing labels.
    :param total_pruning_percentage: The total percentage of weights to prune.
    :param pruning_steps:            Number of pruning steps.
    :param epochs_per_step:          Number of epochs to train each model for in each step.
    :param num_models:               Number of models to create.
    """
    assert pruning_steps > 0, "Pruning steps should be greater than 0"
    assert total_pruning_percentage > 0 and total_pruning_percentage <= 1, "Total pruning percentage should be between 0 and 1"

    # Create the original models if they don't already exist
    create_models(X_train, Y_train, X_test, Y_test, epochs_per_step, num_models)

    feature_shape: tuple[int, ...] = X_train[0].shape
    num_classes: int = 10

    # Iterate over each model
    for model_index in range(num_models):
        iterative_magnitude_pruning(feature_shape, num_classes, model_index, X_train, Y_train, X_test, Y_test, epochs_per_step, total_pruning_percentage, pruning_steps)

