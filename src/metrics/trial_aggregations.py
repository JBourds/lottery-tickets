"""
trial_aggregations.py

Module containing functions for aggregating trial data.

Author: Jordan Bourdeau
Date Created: 5/1/24
"""

import numpy as np

from src.harness import history

def get_sparsity_percentage(trial: history.TrialData) -> float:
    """
    Returns:
        float: Sparsity percentage of a trial.
    """
    return trial.get_sparsity() * 100

def get_early_stopping_iteration(trial: history.TrialData) -> int:
    """
    Returns:
        int: Iteration at which early stopping occurred in a round
            of iterative magnitude pruning.
    """
    return trial.get_early_stopping_iteration()

def get_loss_before_training(trial: history.TrialData) -> float:
    """
    Returns:
        float: Model loss calculated from the masked initial weights.
    """
    return trial.get_loss_before_training()

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

def get_negative_percent_of_weights_across_all_layers(trial: history.TrialData, use_initial: bool = False):
    """
    Returns:
        float: Negative percent of the unpruned initial weights stored before 
            each round of training in a trial across all layers.
    """
    return 100 - get_positive_percent_of_weights_across_all_layers(trial, use_initial=use_initial)
    
def get_positive_percent_of_weights_across_all_layers(trial: history.TrialData, use_initial: bool = False) -> float:
    """
    Function which gets the positive percent of the unpruned initial weights stored
    before each round of training in a trial across all layers.

    Args:
        trial (history.TrialData): Object containing information from a single round
            of iterative pruning.
        use_initial_weights (bool): Flag for whether initial or final weights should be selected.
            Defaults to False and looks at final weights.

    Returns:
        float: Proportion of positive weights across all parameters in the network.
    """
    # Convert masks into boolean Numpy array for indexing, and convert weights into Numpy array as well
    masks: list[np.ndarray] = [mask.numpy().astype('bool') for mask in trial.masks]
    weights: list[np.ndarray] = trial.initial_weights if use_initial else trial.final_weights
    weights = [w.numpy() for w in weights]
    
    # There is a chance weights could be set to 0 but not masked (e.g. bias terms)
    # Because of this, we use the mask as indices and only consider the unmasked portion 
    total_positive: float = np.sum([np.sum(w[mask] >= 0) for mask, w in zip(masks, weights)])
    total_nonzero: float = np.sum([len(w[mask]) for mask, w in zip(masks, weights)])
    
    return (total_positive / total_nonzero) * 100
