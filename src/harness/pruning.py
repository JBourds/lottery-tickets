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

import re
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer

from src.harness import utils

# Typedefs
PruningRule = Callable[[List[Layer], List[Layer], float, Tuple[Any]], None]
SparsityStrategy = Callable[[str], float]


def fast_pruning(layer_name: str) -> float:
    mapping = {
        ('conv.*', 0.15),
        ('dense', 0.3),
    }

    for pattern, pruning in mapping:
        match = re.match(pattern, layer_name, flags=re.IGNORECASE)
        if match:
            return pruning

    return 0


def slow_pruning(layer_name: str) -> float:
    mapping = {
        ('conv.*', 0.05),
        ('dense', 0.1),
    }

    for pattern, pruning in mapping:
        match = re.match(pattern, layer_name, flags=re.IGNORECASE)
        if match:
            return pruning

    return 0


def default_sparsity_strategy(layer_name: str) -> float:
    """
    Default sparsity map which is used for determining pruning rates of layers.
    If no match is found, return 0 (no pruning).

    @param layer_name (str): Name of the Tensorflow layer.

    @returns (float): Percent of remaining parameters to prune.
    """
    mapping = {
        ('conv.*', 0.1),
        ('dense', 0.2),
    }

    for pattern, pruning in mapping:
        match = re.match(pattern, layer_name, flags=re.IGNORECASE)
        if match:
            return pruning

    return 0


def calculate_sparsity(
    layers: List[Layer],
    sparsity_strategy: SparsityStrategy = default_sparsity_strategy,
    sparsity_modifier: float = 1,
) -> float:
    """
    Function used to calculate the next sparsity to pass into a pruning rule
    based on the layers passed into this function and the mapping provided
    between layer names and per-iteration sparsity.

    @param layers (List[Layer]): List of layers to parse. For layerwise pruning
        this will always be a list of 1 element and will just apply the mappings
        from the provided dictionary. For global pruning, this will determine
        the appropriate rate for each layer within it, and calculate the overall
        global sparsity based off it.
    @param sparsity_strategy (SparsityStrategy): Function which maps layer
        names to a sparsity level when pruning.
    @param sparsity_modifier (float): Scaler to modify sparsities with. 
        Currently only used with the last layer.

    @returns (float): New sparsity for the list of layers.
    """
    multipliers = list(
        map(lambda l: 1 - sparsity_strategy(l.name) * sparsity_modifier, layers))
    total_weights = 0
    prunable_weights = 0
    sparsified_weights = 0
    for multiplier, layer in zip(multipliers, layers):
        for w in layer.trainable_weights:
            total_weights += tf.size(w).numpy()
            nonzero = tf.math.count_nonzero(
                w).numpy() if utils.is_prunable(w) else 0
            sparsified_weights += int(multiplier * nonzero)
            prunable_weights += nonzero
    result = sparsified_weights / total_weights
    return result


def prune(
    model: keras.Model,
    mask_model: keras.Model,
    pruning_rule: PruningRule,
    target_sparsity: Callable[[List[Layer]], float],
    *pruning_args,
    global_pruning: bool = False,
) -> None:
    """
    Method which prunes a model's parameters according to some rule
    (e.g. lowest N% magnitudes) and sets their values to 0.

    Also responsible for updating the mask model.

    Args:
        model (keras.Model): Keras model being pruned.
        model (keras.Model): Keras model containing masks to be updated.
        pruning_rule (PruningRule): Function which takes a list of layers as
            input and prunes them according to a predetermined rule.
        target_sparsity: (Callable[[List[Layer]], float]): Function which takes
            a list of layers and outputs the desired level of sparsity to prune.
        *pruning_args: Other positional arguments to pass into the pruning rule along with the
            required target sparsity.
        global_pruning (bool): Boolean flag for if the pruning should be performed globally
            or layerwise.
    """
    # Get all the trainable layers of the model and the corresponding mask layers
    layers_to_prune = [
        layer for layer in model.layers[:-1] if utils.is_prunable(layer)
    ]
    mask_layers = [
        layer for layer in mask_model.layers if utils.is_prunable(layer)
    ]

    if global_pruning:
        pruning_rule(layers_to_prune, mask_layers,
                     target_sparsity(layers_to_prune), *pruning_args)
    else:
        # Prune all the layers until the last one normally
        for layer, masks in zip(layers_to_prune, mask_layers):
            pruning_rule([layer], [masks], target_sparsity(
                [layer]), *pruning_args)

    # Last layer is pruned at half the rate of other layers
    last_layer_sparsity = target_sparsity(
        [model.layers[-1]], sparsity_modifier=0.5)
    pruning_rule([model.layers[-1]], [mask_model.layers[-1]], last_layer_sparsity,
                 *pruning_args)


def low_magnitude_pruning(
    layers: List[Layer],
    mask_layers: List[Layer],
    target_sparsity: float,
):
    magnitude_pruning(layers, mask_layers, target_sparsity, True)


def high_magnitude_pruning(
    layers: List[Layer],
    mask_layers: List[Layer],
    target_sparsity: float,
):
    magnitude_pruning(layers, mask_layers, target_sparsity, False)


def magnitude_pruning(
    layers: List[Layer],
    mask_layers: List[Layer],
    target_sparsity: float,
    prune_low_magnitude: bool,
):
    """
    Function which performs magnitude pruning based on a specified target sparsity
    and strategy (low or high magnitude pruning) and updates the corresponding mask layers.

    NOTE: This function originally also pruned biases but it was later commented
        out since they did not do this in the original paper.

    Args:
        layers (List[Layer]): List of layers to act on.
        mask_layers (List[Layer]): List of masked layers to update with pruning.
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
        weights, biases = layer.get_weights()
        _, biases_mask = mask.get_weights()

        # Get pruned model weights/biases
        if prune_low_magnitude:
            pruned_weights = np.where(
                np.abs(weights) < threshold_weight_bias, 0, weights)
            # pruned_biases = np.where(
            #     np.abs(biases) < threshold_weight_bias, 0, biases)
        else:
            pruned_weights = np.where(
                np.abs(weights) > threshold_weight_bias, 0, weights)
            # pruned_biases = np.where(
            #     np.abs(biases) > threshold_weight_bias, 0, biases)

        # Get new masks after model pruning
        weights_mask = np.where(np.abs(weights) < threshold_weight_bias, 0, 1)
        # biases_mask = np.where(np.abs(biases) < threshold_weight_bias, 0, 1)

        # Update pruned layer weights and masks
        layer.set_weights([pruned_weights, biases])
        mask.set_weights([weights_mask, biases_mask])
