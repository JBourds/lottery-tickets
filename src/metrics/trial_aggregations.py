"""
trial_aggregations.py

Module containing functions for aggregating trial data.

Author: Jordan Bourdeau
Date Created: 5/1/24
"""

from typing import Any, Callable, Generator, List, Union

import numpy as np

from src.harness import architecture
from src.harness import history
from src.metrics import trial_aggregations as t_agg


# ------------------------- Model Information -------------------------

def get_layer_names(trial: history.TrialData) -> List[str]:
    # Temporary for use with TrialData objects that didn't have this
    try:
        arch = trial.architecture
    except Exception as e:
        arch = "conv2"
    return architecture.Architecture.get_model_layers(arch)

def get_dataset(trial: history.TrialData) -> str:
    # Temporary for use with TrialData objects that didn't have this
    try:
        dataset = trial.dataset
    except Exception as e:
        dataset = "cifar"
    return dataset

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
    arr = trial.train_losses if train else trial.validation_losses
    if np.argmax(arr[:--1] == 0) != 0:
        return 0
    arr = arr[arr != 0]
    return np.min(arr)

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


def get_global_average_magnitude(
    trial: history.TrialData,
    use_initial_weights: bool = False,
    use_masked_weights: bool = False,
) -> float:
    """
    Returns:
        float: Average magnitude of unpruned weights across the entire network.
    """
    return _get_average_magnitude(
        trial=trial,
        layerwise=False,
        use_initial_weights=use_initial_weights,
        use_masked_weights=use_masked_weights
    )


def get_layerwise_average_magnitude(
    trial: history.TrialData,
    use_initial_weights: bool = False,
    use_masked_weights: bool = False,
) -> float:
    """
    Returns:
        float: Average magnitude of unpruned weights by layer.
    """
    return _get_average_magnitude(
        trial=trial,
        layerwise=True,
        use_initial_weights=use_initial_weights,
        use_masked_weights=use_masked_weights
    )


def _get_average_magnitude(
    trial: history.TrialData,
    layerwise: bool = False,
    use_initial_weights: bool = False,
    use_masked_weights: bool = False,
) -> Union[float, np.ndarray[float]]:
    """
    Function used to get the average magnitude of parameters, either 
    globally across all layers or layerwise.

    Returns:
        Union[float, np.ndarray[float]]: Single float or list of floats depending on whether it performs
            calculations globally or by layer.
    """
    return _perform_operation_globally_or_layerwise(
        trial=trial,
        operation=_get_average_parameter_magnitude,
        layerwise=layerwise,
        use_initial_weights=use_initial_weights,
        use_masked_weights=use_masked_weights,
    )


def _get_average_parameter_magnitude(weights: List[np.ndarray], masks: List[np.ndarray]) -> float:
    """
    Private function which computes the average magnitude in a list of unmasked weights.

    Args:   
        weights: (List[np.ndarray]): List of Numpy arrays for the weights in each layer being included.
        masks: (List[np.ndarray]): List of Numpy arrays for the masks in each layer being included.

    Returns:
        float: Average parameter magnitude of the unmasked weights.
    """
    assert len(weights) == len(
        masks), 'Weight and mask arrays must be the same length.'
    assert np.all([w.shape == m.shape for w, m in zip(weights, masks)]
                  ), 'Weights and masks must all be the same shape'

    unmasked_weights = [w[mask] for w, mask in zip(weights, masks)]
    unmasked_weight_sum_magnitude = np.sum([np.sum(np.abs(w)) for w in unmasked_weights])
    unmasked_weight_count: int = np.sum([np.sum(w) for w in masks])
    
    if unmasked_weight_count == 0:
        return 0

    return unmasked_weight_sum_magnitude / unmasked_weight_count

# ------------------------- Sign Proportion Metrics -------------------------


def get_global_percent_negative_weights(
    trial: history.TrialData,
    use_initial_weights: bool = False,
    use_masked_weights: bool = False,
) -> float:
    """
    Returns:
        float: Negative percent of the unpruned weights stored across layers in the network.
    """
    return 100 - get_global_percent_positive_weights(
        trial=trial, 
        use_initial_weights=use_initial_weights,
        use_masked_weights=use_masked_weights,
    )


def get_global_percent_positive_weights(
    trial: history.TrialData, 
    use_initial_weights: bool = False,
    use_masked_weights: bool = False,
) -> float:
    """
    Returns:
        float: Positive percent of the unpruned weights stored across layers in the network.
    """
    return 100 * _get_positive_weight_ratio(
        trial=trial, 
        layerwise=False, 
        use_initial_weights=use_initial_weights,
        use_masked_weights=use_masked_weights,
    )


def get_layerwise_percent_negative_weights(
	trial: history.TrialData,
	use_initial_weights: bool = False,
	use_masked_weights: bool = False,
) -> np.ndarray[float]:
    """
    Returns:
        np.ndarray[float]: Negative percent of the unpruned weights stored by layer in the network.
    """
    return 100 - get_layerwise_percent_positive_weights(
        trial=trial, 
        use_initial_weights=use_initial_weights,
        use_masked_weights=use_masked_weights,
    )


def get_layerwise_percent_positive_weights(
	trial: history.TrialData,
	use_initial_weights: bool = False,
	use_masked_weights: bool = False,
) -> np.ndarray[float]:
    """
    Returns:
        np.ndarray[float]: Positive percent of the unpruned weights stored by layer in the network.
    """
    return 100 * _get_positive_weight_ratio(
        trial=trial, 
        layerwise=True, 
        use_initial_weights=use_initial_weights,
        use_masked_weights=use_masked_weights,
    )


def _get_positive_weight_ratio(
    trial: history.TrialData,
    layerwise: bool = False,
    use_initial_weights: bool = False,
    use_masked_weights: bool = False,
) -> Union[float, np.ndarray[float]]:
    """
    Function used to get the positive weight ratio, either globally across all layers or
    layerwise.

    Returns:
        Union[float, np.ndarray[float]]: Single float or list of floats depending on whether it performs
            calculations globally or by layer.
    """
    return _perform_operation_globally_or_layerwise(
        trial=trial,
        operation=_get_ratio_of_unmasked_positive_weights,
        layerwise=layerwise,
        use_initial_weights=use_initial_weights,
        use_masked_weights=use_masked_weights,
    )

def _get_ratio_of_unmasked_positive_weights(weights: List[np.ndarray], masks: List[np.ndarray]) -> float:
    """
    Private function which computes the proportion of unmasked positive weights.

    Args:   
        weights: (List[np.ndarray]): List of Numpy arrays for the weights in each layer being included.
        masks: (List[np.ndarray]): List of Numpy arrays for the masks in each layer being included.

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
    
    if total_nonzero == 0:
        return 1

    return total_positive / total_nonzero

# ------------------------- Aggregate Over Trials -------------------------


def aggregate_across_trials(
    experiments: List[Generator[history.TrialData, None, None]],
    trial_aggregations: List[Callable[[history.TrialData], Any]],
) -> List[List[Any]]:
    """
    Method used to aggregate over all the trials within a summary
    using a user-defined function to aggregate trial data.

    @param experiments (List[Generator[history.TrialData, None, None]]): List
        containing generator functions for every experiment where the generators
        produce trial data objects one at a time. This minimizes the amount of data
        which needs to be in memory at once.
    @param trial_aggreagtions (List[Callable[[history.TrialData], Any]]): Functions which
        takes in a single trial's data and computes an aggregate statistic from it.

    @returns List[List[List[Any]]]): A `N` x `M` x `A`  array (list) where `A` is the number of aggregations,
        `N` is the number of experiments and `M` corresponds to the number of trials in each experiment
        (required # pruning steps + 1).
    """
    aggregated_data = [[] for _ in trial_aggregations]
    for e_index, experiment in enumerate(experiments):
        for t_index, trial in enumerate(experiment):
            for agg_index, aggregation in enumerate(trial_aggregations):
                if t_index == 0:
                    aggregated_data[agg_index].append([])
                aggregated_data[agg_index][e_index].append(aggregation(trial))

    return aggregated_data

# ------------------------- Private Helper Methods -------------------------


def _perform_operation_globally_or_layerwise(
    trial: history.TrialData,
    operation: callable,
    layerwise: bool = False,
    use_initial_weights: bool = False,
    use_masked_weights: bool = False,
) -> any:
    """
    Function used to get the average magnitude of parameters, either 
    globally across all layers or layerwise.

    Args:
        operation (callable): Function which is being performed layerwise of globally.
        layerwise (bool, optional): Get proportion for each layer. Defaults to False.
        use_initial_weights (bool, optional): Use initial, masked weights or final trained weights. 
            Defaults to False.
        use_masked_weights (Optional[bool]): Flag for whether operation should be performed on the
            masked weights instead (just flips boolean array).

    Returns:
        any: Returns type depends on the operation.
    """
    # Convert masks into boolean Numpy array for indexing, and convert weights into Numpy array as well
    masks = [np.logical_not(mask) if use_masked_weights else mask.astype(bool) for mask in trial.masks]
    weights = trial.initial_weights if use_initial_weights else trial.final_weights
    weights = [w for w in weights]

    if layerwise:
        # Since the private method expects a list, just make each entry into a list of 1 element
        masks = [[np.logical_not(mask) if use_masked_weights else mask.astype(bool)] for mask in masks]
        weights = [[w] for w in weights]
        return np.array([operation(w, m) for w, m in zip(weights, masks)])

    return operation(weights, masks)
