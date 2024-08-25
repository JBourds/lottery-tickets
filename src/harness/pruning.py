"""
pruning.py

Module containing functions to prune model weights.

Interface specs:

Function used to prune the model is `prune` but the pruning rule/strategy can be supplied
within the method used to run the experiment, and any additional pruning arguments will
be unpacked into the pruning rule function call.

The requirements for any pruning rule method is as follows:
    - Pruning rules must be able to take 3 arguments:
        > List of keras model layers being pruned
        > List of keras model mask layers to also prune
        > Target sparsity
    
      
Author: Jordan Bourdeau
Date Created: 3/24/24
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.harness import utils


def get_sparsity_percents(
    model: keras.Model,
    first_step_pruning_percent: float,
    target_sparsity: float,
) -> list[float]:
    """
    Function to get arrays of model sparsity at each step of pruning based on a constant pruning %
    applied to nonzer-parameters.

    Args:
        model (keras.Model): Keras model to get the percents for.
        first_step_pruning_percent (float): Initial pruning percent step to use in subsequent steps.
        target_sparsity (float): Desired level of sparsity.

    Returns:
        list[float]: List of pruning percents corresponding to each step of iterative magnitude pruning.
    """

    def total_sparsity(
        original_weight_counts: list[int],
        current_weight_counts: list[int]
    ) -> float:

        return np.sum(current_weight_counts) / np.sum(original_weight_counts)

    def sparsify(
        original_weight_counts: list[int],
        current_weight_counts: list[int],
        original_pruning_percent: float,
    ) -> float:
        """
        Function to calculate new sparsity percentage and update weight counts.

        Args:
            original_weight_counts (list[int]): Original nonzero parameter counts.
            current_weight_counts (list[int]): Current nonzero parameter counts.
            original_pruning_percent (float): Original pruning % used on the first step.

        Returns:
            float: Pruning % of previous step.
        """
        sparsities = []
        for idx, (original, current) in enumerate(zip(original_weight_counts, current_weight_counts)):
            if original == 0:
                continue
            new_weight_count = np.round(
                current * (1 - original_pruning_percent))
            sparsities.append(new_weight_count / original)
            current_weight_counts[idx] = new_weight_count
        return np.round(np.mean(sparsities), decimals=5)

    layer_weight_counts = utils.get_layer_weight_counts(model)
    sparsities = [1]   # First iteration will start at 100% parameters

    # Elementwise copy
    current_weight_counts = [
        weight_count for weight_count in layer_weight_counts]

    while total_sparsity(layer_weight_counts, current_weight_counts) > target_sparsity:
        sparsities.append(sparsify(layer_weight_counts,
                          current_weight_counts, first_step_pruning_percent))

    return sparsities


def prune(
    model: keras.Model,
    mask_model: keras.Model,
    pruning_rule: callable,
    target_sparsity: float,
    *pruning_args,
    global_pruning: bool = False,
) -> list[np.ndarray]:
    """
    Method which prunes a model's parameters according to some rule 
    (e.g. lowest N% magnitudes) and sets their values to 0.

    Also responsible for updating the mask model.

    Args:
        model (keras.Model): Keras model being pruned.
        model (keras.Model): Keras model containing masks to be updated.
        pruning_rule (callable): Function which takes a list of layers as input and prunes them
            according to a predetermined rule.
        target_sparsity (float): Target sparsity for the pruning method.
        *pruning_args: Other positional arguments to pass into the pruning rule along with the
            required target sparsity.
        global_pruning (bool): Boolean flag for if the pruning should be performed globally
            or layerwise.

    Returns:
        List of Numpy arrays with indices of pruned weights.
    """
    # Get all the trainable layers of the model and the corresponding mask layers
    layers_to_prune = [
        layer for layer in model.layers if utils.is_prunable(layer)]
    mask_layers = [
        layer for layer in mask_model.layers if utils.is_prunable(layer)]

    if global_pruning:
        return pruning_rule(layers_to_prune, mask_layers, target_sparsity, *pruning_args)
    else:
        # Prune all the layers until the last one normally
        for layer, masks in zip(layers_to_prune[:-1], mask_layers[:-1]):
            pruning_rule([layer], [masks], target_sparsity, *pruning_args)

        # Last layer is pruned at half the rate of other layers
        last_layer_sparsity = (1 + target_sparsity) / 2
        pruning_rule([layers_to_prune[-1]], [mask_layers[-1]],
                     last_layer_sparsity, *pruning_args)


def low_magnitude_pruning(
    layers: list[keras.layers.Layer],
    mask_layers: list[keras.layers.Layer],
    target_sparsity: float,
):
    magnitude_pruning(layers, mask_layers, target_sparsity, True)


def high_magnitude_pruning(
    layers: list[keras.layers.Layer],
    mask_layers: list[keras.layers.Layer],
    target_sparsity: float,
):
    magnitude_pruning(layers, mask_layers, target_sparsity, False)


def magnitude_pruning(
    layers: list[keras.layers.Layer],
    mask_layers: list[keras.layers.Layer],
    target_sparsity: float,
    prune_low_magnitude: bool,
):
    """
    Function which performs magnitude pruning based on a specified target sparsity
    and strategy (low or high magnitude pruning) and updates the corresponding mask layers.

    Args:
        layers (list[keras.layers.Layer]): List of layers to act on.
        mask_layers (list[keras.layers.Layer]): List of masked layers to update with pruning.
        pruning_percentage (float): Target sparsity of the model (% of nonzero weights remaining).
        prune_low_magnitude (bool): Flag for whether to perform low or high magnitude pruning.
    """
    # Skip if we don't actually need to prune
    if target_sparsity == 1:
        return
    elif target_sparsity > 1 or target_sparsity < 0:
        raise ValueError(
            f'Found target sparsity of {target_sparsity} but must be between 0 and 1')

    # Calculate the number of weights to prune for each layer
    total_params = sum(np.prod(layer.get_weights()[
                       0].shape) + np.prod(layer.get_weights()[1].shape) for layer in layers)
    num_params_to_prune = int(total_params * (1 - target_sparsity))

    # Flatten and sort the weights and biases across all layers
    all_weights_biases = [np.concatenate([layer.get_weights()[0].flatten(
    ), layer.get_weights()[1].flatten()]) for layer in layers]
    all_weights_biases = np.concatenate(all_weights_biases)
    sorted_weights_biases = np.sort(np.abs(all_weights_biases))

    # Find the threshold weight value for pruning
    index = num_params_to_prune if prune_low_magnitude else -num_params_to_prune
    threshold_weight_bias = sorted_weights_biases[index]

    # Apply pruning by setting low or high magnitude weights to zero
    for layer, mask in zip(layers, mask_layers):
        weights = layer.get_weights()[0]
        biases = layer.get_weights()[1]

        # Get pruned model weights/biases
        if prune_low_magnitude:
            pruned_weights = np.where(
                np.abs(weights) < threshold_weight_bias, 0, weights)
            pruned_biases = np.where(
                np.abs(biases) < threshold_weight_bias, 0, biases)
        else:
            pruned_weights = np.where(
                np.abs(weights) > threshold_weight_bias, 0, weights)
            pruned_biases = np.where(
                np.abs(biases) > threshold_weight_bias, 0, biases)

        # Get new masks after model pruning
        weights_mask = np.where(np.abs(weights) < threshold_weight_bias, 0, 1)
        biases_mask = np.where(np.abs(biases) < threshold_weight_bias, 0, 1)

        # Update pruned layer weights and masks
        layer.set_weights([pruned_weights, pruned_biases])
        mask.set_weights([weights_mask, biases_mask])

