"""
pruning.py

Module containing functions to prune model weights.

Interface:

Function used to prune the model is `prune` but the pruning rule/strategy can be supplied
within the method used to run the experiment. The only argument which will be passed to
the pruning rule function within the training loop is the target sparsity.

The requirements for any pruning rule method is as follows:
    - First argument is the keras model being pruned
    - All arguments other than the target sparsity for an iteration are initialized
      beforehand (e.g. by using partial function application).
    - The function has *args in its function definition.

Create By: Jordan Bourdeau
Date Created: 3/24/24
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.harness import utils

def get_sparsity_percents(
    model: keras.Model,
    first_step_pruning_percent: float,
    target_sparsity: float
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
        """
        Helper function to calculate total sparsity of parameters.

        Args:
            original_weight_counts (list[int]): Initial nonzero weight counts.
            current_weight_counts (list[int]): Current nonzero weights counts at this step of pruning.

        Returns:
            float: Total sparsity across all weights.
        """
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
        sparsities: list[float] = []
        for idx, (original, current) in enumerate(zip(original_weight_counts, current_weight_counts)):
            if original == 0:
                continue
            new_weight_count: int = np.round(current * (1 - original_pruning_percent))
            sparsities.append((original - new_weight_count) / original)
            current_weight_counts[idx] = new_weight_count
        return np.round(np.mean(sparsities), decimals=5)
    
    layer_weight_counts: list[int] = utils.get_layer_weight_counts(model)
    sparsities: list[float] = [0]   # First iteration will start at 100% parameters
    
    # Elementwise copy
    current_weight_counts: list[int] = [weight_count for weight_count in layer_weight_counts]
    
    while total_sparsity(layer_weight_counts, current_weight_counts) > target_sparsity:
        sparsities.append(sparsify(layer_weight_counts, current_weight_counts, first_step_pruning_percent))

    return sparsities

def update_masks(pruned_model: keras.Model, mask_model: keras.Model):
    """
    Function responsible for updating the mask model to reflect the pruned model.

    Args:
        pruned_model (keras.Model): Model which has had parameters pruned.
        mask_model (keras.Model): Model whose weights correspond to masks.
    """
    for pruned_layer, mask_layer in zip(pruned_model.layers, mask_model.layers):
        weights, biases = pruned_layer.get_weights()
        mask: tf.Tensor = tf.where(tf.equal(weights, 0), tf.zeros_like(weights), tf.ones_like(weights))
        # Update the weights, but always keep the bias masks to 1
        mask_layer.set_weights([mask, tf.ones_like(biases)])

def prune(model: keras.Model, pruning_rule: callable, *pruning_rule_args, global_pruning: bool = False):
    """
    Method which prunes a model's parameters according to some rule 
    (e.g. lowest N% magnitudes) and sets their values to 0.

    Args:
        model (Model): Keras model being pruned.
        pruning_rule (callable): Function which takes a list of layers as input and prunes them
            according to a predetermined rule.
        *pruning_rule_args: Arguments to be passed into the pruning rule.
        global_pruning (bool): Boolean flag for if the pruning should be performed globally
            or layerwise.
    """
    # Get all the trainable layers of the model
    layers_to_prune = [layer for layer in model.layers if utils.is_prunable(layer)]

    if global_pruning:
        pruning_rule(layers_to_prune, *pruning_rule_args)
    else:
        for layer in layers_to_prune:
            pruning_rule([layer], *pruning_rule_args)
        
def constant_value_pruning(layers: list[keras.layers.Layer], value: int = 0, *args):
    """
    Function which is used to prune weights to a constant value.

    Args:
        layers (list[keras.layers.Layer]): List of layers to act on.
        value (int, optional): Value to set pruned weights to. Defaults to 0.
    """
    for layer in layers:
        layer.set_weights([weights * 0 for weights in layer.get_weights()])
        
def low_magnitude_pruning(layers: list[keras.layers.Layer], target_sparsity: float):
    """
    Function which performs low magnitude pruning based on a specified target sparsity.

    Args:
        layers (list[keras.layers.Layer]): List of layers to act on.
        pruning_percentage (float): Target sparsity of the model (% of nonzero weights remaining).
    """
    
    # Skip if we don't actually need to prune
    if target_sparsity == 1:
        return
    elif target_sparsity > 1 or target_sparsity < 0:
        raise ValueError(f'Found target sparsity of {target_sparsity} but must be between 0 and 1')

    # Calculate the number of weights to prune for each layer
    total_params: int = sum(np.prod(layer.get_weights()[0].shape) + np.prod(layer.get_weights()[1].shape) for layer in layers)
    num_params_to_prune: int = int(total_params * target_sparsity)

    # Flatten and sort the weights and biases across all layers
    all_weights_biases: list[np.ndarray] = [np.concatenate([layer.get_weights()[0].flatten(), layer.get_weights()[1].flatten()]) for layer in layers]
    all_weights_biases = np.concatenate(all_weights_biases)
    sorted_weights_biases: np.ndarray = np.sort(np.abs(all_weights_biases))

    # Find the threshold weight value for pruning
    threshold_weight_bias = sorted_weights_biases[num_params_to_prune]

    # Apply pruning by setting low magnitude weights to zero
    for layer in layers:
        weights = layer.get_weights()[0]
        biases = layer.get_weights()[1]
        pruned_weights = np.where(np.abs(weights) < threshold_weight_bias, 0.0, weights)
        pruned_biases = np.where(np.abs(biases) < threshold_weight_bias, 0.0, biases)
        layer.set_weights([pruned_weights, pruned_biases])
        