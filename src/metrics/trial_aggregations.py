"""
trial_aggregations.py

Module containing functions for aggregating trial data.

Author: Jordan Bourdeau
Date Created: 5/1/24
"""

import numpy as np

from src.harness import history

# ------------------------- Sparsity -------------------------

def get_sparsity_percentage(trial: history.TrialData) -> float:
    """
    Returns:
        float: Sparsity percentage of a trial.
    """
    return trial.get_sparsity() * 100

# ------------------------- Training Time Metrics -------------------------

def get_early_stopping_iteration(trial: history.TrialData) -> int:
    """
    Returns:
        int: Iteration at which early stopping occurred in a round
            of iterative magnitude pruning.
    """
    return trial.get_early_stopping_iteration()

# ------------------------- Loss Metrics -------------------------

def get_loss_before_training(trial: history.TrialData) -> float:
    """
    Returns:
        float: Model loss calculated from the masked initial weights.
    """
    return trial.get_loss_before_training()

def get_best_loss(trial: history.TrialData, train: bool = False) -> float:
  """
  Returns:
      float: Best model accuracy from within a round of iterative pruning.
  """
  return np.max(trial.train_losses if train else trial.test_losses)

# ------------------------- Accuracy Metrics -------------------------

def get_accuracy_before_training(trial: history.TrialData) -> float:
    """
    Returns:
        float: Model accuracy calculated from the masked initial weights.
    """
    return trial.get_accuracy_before_training()

def get_best_accuracy_percent(trial: history.TrialData, train: bool = False) -> float:
    """
    Returns:
        float: Best model accuracy from within a round of iterative pruning.
    """
    return np.max(trial.train_accuracies if train else trial.test_accuracies) * 100

# ------------------------- Magnitude Metrics -------------------------

def get_global_average_magnitude(trial: history.TrialData, use_initial_weights: bool = False) -> float:
    """
    Returns:
        float: Average magnitude of unpruned weights across the entire network.
    """
    return trial.get_average_magnitude(layerwise=False, use_initial_weights=use_initial_weights)

def get_layerwise_average_magnitude(trial: history.TrialData, use_initial_weights: bool = False) -> float:
  """
  Returns:
      float: Average magnitude of unpruned weights by layer.
  """
  return trial.get_average_magnitude(layerwise=True, use_initial_weights=use_initial_weights)


# ------------------------- Sign Proportion Metrics -------------------------

def get_global_percent_negative_weights(trial: history.TrialData, use_initial_weights: bool = False) -> float:
    """
    Returns:
        float: Negative percent of the unpruned weights stored across layers in the network.
    """
    return 100 - get_global_percent_positive_weights(trial, use_initial_weights=use_initial_weights)
    
def get_global_percent_positive_weights(trial: history.TrialData, use_initial_weights: bool = False) -> float:
    """
    Returns:
        float: Positive percent of the unpruned weights stored across layers in the network.
    """
    return trial.get_positive_weight_ratio(layerwise=False, use_initial_weights=use_initial_weights) * 100

def get_layerwise_percent_negative_weights(trial: history.TrialData, use_initial_weights: bool = False) -> float:
    """
    Returns:
        list[float]: Negative percent of the unpruned weights stored by layer in the network.
    """
    return 100 - get_layerwise_percent_positive_weights(trial, use_initial_weights=use_initial_weights)
    
def get_layerwise_percent_positive_weights(trial: history.TrialData, use_initial_weights: bool = False) -> float:
    """
    Returns:
        list[float]: Positive percent of the unpruned weights stored by layer in the network.
    """
    return trial.get_positive_weight_ratio(layerwise=True, use_initial_weights=use_initial_weights) * 100