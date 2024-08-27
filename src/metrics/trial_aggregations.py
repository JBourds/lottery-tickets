"""
trial_aggregations.py

Module containing functions for aggregating trial data.

Author: Jordan Bourdeau
Date Created: 5/1/24
"""

from typing import Union

import numpy as np

from src.harness import history
from src.metrics import trial_aggregations as t_agg

# ------------------------- Sparsity -------------------------


def get_pruning_step(trial: history.TrialData) -> int:
    """
    Get the pruning step of the TrialData.

    Returns: 
        int: The trials respective pruning step.
    """
    return trial.pruning_step


def get_sparsity(trial: history.TrialData) -> float:
    """
    Calculate the sparsity of the model at this particular training round.

    Returns:
        float: The sparsity ratio of enabled parameters to total parameters.
    """
    enabled_parameter_count: int = np.sum(
        [np.sum(mask) for mask in trial.masks])
    total_parameter_count: int = np.sum(
        [np.size(mask) for mask in trial.masks])
    return enabled_parameter_count / total_parameter_count


def get_sparsity_percentage(trial: history.TrialData) -> float:
    """
    Returns:
        float: Sparsity percentage of a trial.
    """
    return get_sparsity(trial) * 100

# ------------------------- Training Time Metrics -------------------------


def get_early_stopping_iteration(trial: history.TrialData) -> int:
    """
    Get the step at which early stopping occurred during training.

    Returns:
        int: The step at which training was stopped early.
    """
    performance_evaluation_frequency: int = trial.train_accuracies.shape[
        0] // trial.validation_accuracies.shape[0]
    nonzero_indices = np.nonzero(trial.train_accuracies == 0)[0]
    stop_index: int = len(trial.train_accuracies) if len(
        nonzero_indices) == 0 else nonzero_indices[0]
    return stop_index * performance_evaluation_frequency

# ------------------------- Loss Metrics -------------------------


def get_loss_before_training(trial: history.TrialData) -> float:
    """
    Returns:
        float: Model loss on the masked initial weights.
    """
    return trial.loss_before_training


def get_best_loss(trial: history.TrialData, train: bool = False) -> float:
    """
    Returns:
        float: Best model accuracy from within a round of iterative pruning.
    """
    return np.min(trial.train_losses if train else trial.validation_losses)

# ------------------------- Accuracy Metrics -------------------------


def get_accuracy_before_training(trial: history.TrialData) -> float:
    """
    Returns:
        float: Model accuracy calculated from the masked initial weights.
    """
    return trial.accuracy_before_training


def get_best_accuracy_percent(trial: history.TrialData, train: bool = False) -> float:
    """
    Returns:
        float: Best model accuracy from within a round of iterative pruning.
    """
    return np.max(trial.train_accuracies if train else trial.validation_accuracies) * 100

# ------------------------- Magnitude Metrics -------------------------


def get_global_average_magnitude(trial: history.TrialData, use_initial_weights: bool = False) -> float:
    """
    Returns:
        float: Average magnitude of unpruned weights across the entire network.
    """
    return _get_average_magnitude(trial=trial, layerwise=False, use_initial_weights=use_initial_weights)


def get_layerwise_average_magnitude(trial: history.TrialData, use_initial_weights: bool = False) -> float:
    """
    Returns:
        float: Average magnitude of unpruned weights by layer.
    """
    return _get_average_magnitude(trial=trial, layerwise=True, use_initial_weights=use_initial_weights)


def _get_average_magnitude(
    trial: history.TrialData,
    layerwise: bool = False,
    use_initial_weights: bool = False,
) -> Union[float, list[float]]:
    """
    Function used to get the average magnitude of parameters, either 
    globally across all layers or layerwise.

    Returns:
        Union[float, list[float]]: Single float or list of floats depending on whether it performs
            calculations globally or by layer.
    """
    return _perform_operation_globally_or_layerwise(
        trial=trial,
        operation=_get_average_parameter_magnitude,
        layerwise=layerwise,
        use_initial_weights=use_initial_weights
    )


def _get_average_parameter_magnitude(weights: list[np.ndarray], masks: list[np.ndarray]) -> float:
    """
    Private function which computes the average magnitude in a list of unmasked weights.

    Args:   
        weights: (list[np.ndarray]): List of Numpy arrays for the weights in each layer being included.
        masks: (list[np.ndarray]): List of Numpy arrays for the masks in each layer being included.

    Returns:
        float: Average parameter magnitude of the unmasked weights.
    """
    assert len(weights) == len(
        masks), 'Weight and mask arrays must be the same length.'
    assert np.all([w.shape == m.shape for w, m in zip(weights, masks)]
                  ), 'Weights and masks must all be the same shape'

    unmasked_weights: list[np.ndarray] = [w[mask]
                                          for w, mask in zip(weights, masks)]
    unmasked_weight_sum_magnitude: float = np.sum(
        [np.sum(np.abs(w)) for w in unmasked_weights])
    unmasked_weight_count: int = np.sum([np.size(w) for w in unmasked_weights])

    return unmasked_weight_sum_magnitude / unmasked_weight_count

# ------------------------- Sign Proportion Metrics -------------------------


def get_global_percent_negative_weights(trial: history.TrialData, use_initial_weights: bool = False) -> float:
    """
    Returns:
        float: Negative percent of the unpruned weights stored across layers in the network.
    """
    return 100 - get_global_percent_positive_weights(trial=trial, use_initial_weights=use_initial_weights)


def get_global_percent_positive_weights(trial: history.TrialData, use_initial_weights: bool = False) -> float:
    """
    Returns:
        float: Positive percent of the unpruned weights stored across layers in the network.
    """
    return _get_positive_weight_ratio(trial=trial, layerwise=False, use_initial_weights=use_initial_weights) * 100


def get_layerwise_percent_negative_weights(trial: history.TrialData, use_initial_weights: bool = False) -> float:
    """
    Returns:
        list[float]: Negative percent of the unpruned weights stored by layer in the network.
    """
    return 100 - get_layerwise_percent_positive_weights(trial=trial, use_initial_weights=use_initial_weights)


def get_layerwise_percent_positive_weights(trial: history.TrialData, use_initial_weights: bool = False) -> float:
    """
    Returns:
        list[float]: Positive percent of the unpruned weights stored by layer in the network.
    """
    return _get_positive_weight_ratio(trial=trial, layerwise=True, use_initial_weights=use_initial_weights) * 100


def _get_positive_weight_ratio(
    trial: history.TrialData,
    layerwise: bool = False,
    use_initial_weights: bool = False
) -> Union[float, list[float]]:
    """
    Function used to get the positive weight ratio, either globally across all layers or
    layerwise.

    Returns:
        Union[float, list[float]]: Single float or list of floats depending on whether it performs
            calculations globally or by layer.
    """
    return _perform_operation_globally_or_layerwise(
        trial=trial,
        operation=_get_ratio_of_unmasked_positive_weights,
        layerwise=layerwise,
        use_initial_weights=use_initial_weights
    )


def _get_ratio_of_unmasked_positive_weights(weights: list[np.ndarray], masks: list[np.ndarray]) -> float:
    """
    Private function which computes the proportion of unmasked positive weights.

    Args:   
        weights: (list[np.ndarray]): List of Numpy arrays for the weights in each layer being included.
        masks: (list[np.ndarray]): List of Numpy arrays for the masks in each layer being included.

    Returns:
        float: Proportion of postive parameters in the unmasked weights.
    """
    assert len(weights) == len(
        masks), 'Weight and mask arrays must be the same length.'
    assert np.all([w.shape == m.shape for w, m in zip(weights, masks)]
                  ), 'Weights and masks must all be the same shape'

    # There is a chance weights could be set to 0 but not masked (e.g. bias terms)
    # Because of this, we use the mask as indices and only consider the unmasked portion
    total_positive: float = np.sum(
        [np.sum(w[mask] >= 0) for w, mask in zip(weights, masks)])
    total_nonzero: float = np.sum([np.size(w[mask])
                                  for w, mask in zip(weights, masks)])

    return total_positive / total_nonzero

# ------------------------- Density -------------------------


def _get_weight_density(
    trial: history.TrialData,
    layerwise: bool = False,
    use_initial_weights: bool = False,
) -> Union[float, list[float]]:
    """
    Function used to get a density plot of weight distributions,

    Returns:
        Union[float, list[float]]: Single float or list of floats depending on whether it performs
            calculations globally or by layer.
    """
    return _perform_operation_globally_or_layerwise(
        trial=trial,
        operation=_get_weight_density,
        layerwise=layerwise,
        use_initial_weights=use_initial_weights
    )


def _get_weight_density(weights: list[np.ndarray], masks: list[np.ndarray]) -> np.array:
    """
    Private function which computes a density plot of weights.

    Args:   
        weights: (list[np.ndarray]): List of Numpy arrays for the weights in each layer being included.
        masks: (list[np.ndarray]): List of Numpy arrays for the masks in each layer being included.

    Returns:
        np.array[float]: Density plot of weights.
    """
    assert len(weights) == len(
        masks), 'Weight and mask arrays must be the same length.'
    assert np.all([w.shape == m.shape for w, m in zip(weights, masks)]
                  ), 'Weights and masks must all be the same shape'

    # Flatten the weights according to the masks
    unmasked_weights = np.concatenate(
        [w[mask].flatten() for w, mask in zip(weights, masks)])

    # Compute the density plot of target weights
    density, bins = np.histogram(unmasked_weights, bins=20, density=True)

    return density, bins

# ------------------------- Aggregate Over Trials -------------------------


def aggregate_across_trials(
    summary: history.ExperimentSummary,
    trial_aggregation: callable,
) -> list[list]:
    """
    Method used to aggregate over all the trials within a summary
    using a user-defined function to aggregate trial data.

    Args:
        summary (callable): `ExperimentSummary` object being aggregated over.
        trial_aggregation (Callable): Function which returns a single value when
            called on a `TrialData` object.

    Returns:
        list[list]: A 2D list where each inner list contains the aggregated values
            gathered from every trial object.

            e.g.

            [
                [2.5, 2.3, ..., 2.5],   # All aggregated values gathered from trial index = 0
                [2.5, 2.3, ..., 2.5],   # All aggregated values gathered from trial index = 1
                ...
            ]
    """
    trials_aggregated: dict = {}
    # Iterate across experiments
    for experiment in summary.experiments.values():
        # Iterate over trials within an experiment
        for trial in experiment.trials.values():
            pruning_step: int = t_agg.get_pruning_step(trial)
            if pruning_step in trials_aggregated.keys():
                trials_aggregated.get(pruning_step).append(
                    (trial_aggregation(trial)))
            else:
                trials_aggregated[pruning_step] = [trial_aggregation(trial)]

    return list(trials_aggregated.values())

# ------------------------- Private Helper Methods -------------------------


def _perform_operation_globally_or_layerwise(
    trial: history.TrialData,
    operation: callable,
    layerwise: bool = False,
    use_initial_weights: bool = False,
) -> any:
    """
    Function used to get the average magnitude of parameters, either 
    globally across all layers or layerwise.

    Args:
        operation (callable): Function which is being performed layerwise of globally.
        layerwise (bool, optional): Get proportion for each layer. Defaults to False.
        use_initial_weights (bool, optional): Use initial, masked weights or final trained weights. 
            Defaults to False.

    Returns:
        any: Returns type depends on the operation.
    """
    # Convert masks into boolean Numpy array for indexing, and convert weights into Numpy array as well
    masks: list[np.ndarray] = [mask.astype(bool) for mask in trial.masks]
    weights: list[np.ndarray] = trial.initial_weights if use_initial_weights else trial.final_weights
    weights = [w for w in weights]

    if layerwise:
        # Since the private method expects a list, just make each entry into a list of 1 element
        masks = [[m] for m in masks]
        weights = [[w] for w in weights]
        return list(map(operation, zip(weights, masks)))

    return operation(weights, masks)
