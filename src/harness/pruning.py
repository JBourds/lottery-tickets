"""
pruning.py

Module containing function to prune masks based on magnitude.

Create By: Jordan Bourdeau
Date Created: 3/24/24
"""

import numpy as np
import tensorflow as tf
from tensorflow_model_optimization.sparsity import keras as sparsity

# Define pruning parameters and callback
def create_pruning_parameters(target_sparsity: float, begin_Step: int, end_step: int, frequency: int) -> dict:
    """
    Create the dictionary of pruning parameters to be used.
    """
    return {
        'pruning_schedule': sparsity.ConstantSparsity(
            target_sparsity=target_sparsity, 
            begin_step=begin_Step,
            end_step=end_step, 
            frequency=frequency
        )
    }

def create_pruning_callback(monitor: str, patience: int, minimum_delta: float) -> list:
    """
    Create a callback to be performed during pruning.
    """
    return [
        sparsity.UpdatePruningStep(),
        # sparsity.PruningSummaries(log_dir = logdir, profile_batch=0),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, 
            patience=patience,
            min_delta=minimum_delta
        )
    ]

# For each layer, there are synaptic connections from the previous layer and the neurons
def get_layer_weight_counts(model: tf.keras.Model) -> list[int]:
    """
    Function to return a list of integer values for the number of 
    parameters in each layer.
    """
    def get_num_layer_weights(layer: tf.keras.layers.Layer) -> int:
        layer_weight_count: int = 0
        weights: list[np.array] = layer.get_weights()

        for idx in range(len(weights))[::2]:
            synapses: np.ndarray = weights[idx]
            neurons: np.array = weights[idx + 1]
            layer_weight_count += np.prod(synapses.shape) + np.prod(neurons.shape)

        return layer_weight_count
    
    return list(map(get_num_layer_weights, model.layers))

def get_pruning_percents(
        layer_weight_counts: list[int], 
        first_step_pruning_percent: float,
        target_sparsity: float
        ) -> list[np.array]:
    """
    Function to get arrays of model sparsity at each step of pruning.
    """

    def total_sparsity(
            original_weight_counts: list[int], 
            current_weight_counts: list[int]
            ) -> float:
        """
        Helper function to calculate total sparsity of parameters.
        """
        return np.sum(current_weight_counts) / np.sum(original_weight_counts)
    
    def sparsify(
            original_weight_counts: list[int], 
            current_weight_counts: list[int], 
            original_pruning_percent: float
            ) -> list[float]:
        sparsities: list[float] = []
        for idx, (original, current) in enumerate(zip(original_weight_counts, current_weight_counts)):
            if original == 0:
                continue
            new_weight_count: int = np.round(current * (1 - original_pruning_percent))
            sparsities.append((original - new_weight_count) / original)
            current_weight_counts[idx] = new_weight_count
        return np.round(np.mean(sparsities), decimals=5)
    
    sparsities: list[float] = []
    
    # Elementwise copy
    current_weight_counts: list[int] = [weight_count for weight_count in layer_weight_counts]
    
    while total_sparsity(layer_weight_counts, current_weight_counts) > target_sparsity:
        sparsities.append(sparsify(layer_weight_counts, current_weight_counts, first_step_pruning_percent))

    return sparsities
