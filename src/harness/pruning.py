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
            sparsities.append(new_weight_count / original)
            current_weight_counts[idx] = new_weight_count
        return np.round(np.mean(sparsities), decimals=5)
    
    layer_weight_counts: list[int] = utils.get_layer_weight_counts(model)
    sparsities: list[float] = [1]   # First iteration will start at 100% parameters
    
    # Elementwise copy
    current_weight_counts: list[int] = [weight_count for weight_count in layer_weight_counts]
    
    while total_sparsity(layer_weight_counts, current_weight_counts) > target_sparsity:
        sparsities.append(sparsify(layer_weight_counts, current_weight_counts, first_step_pruning_percent))

    return sparsities

def prune(
    model: keras.Model, 
    mask_model: keras.Model,
    pruning_rule: callable, 
    target_sparsity: float, 
    global_pruning: bool = False) -> list[np.ndarray]:
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
        global_pruning (bool): Boolean flag for if the pruning should be performed globally
            or layerwise.
            
    Returns:
        List of Numpy arrays with indices of pruned weights.
    """
    # Get all the trainable layers of the model and the corresponding mask layers
    layers_to_prune: list[keras.layers.Layer] = [layer for layer in model.layers if utils.is_prunable(layer)]
    mask_layers: list[keras.layers.Layer] = [layer for layer in mask_model.layers if utils.is_prunable(layer)]

    if global_pruning:
        return pruning_rule(layers_to_prune, mask_layers, target_sparsity)
    else:
        # Prune all the layers until the last one normally
        for layer, masks in zip(layers_to_prune[:-1], mask_layers[:-1]):
            pruning_rule([layer], [masks], target_sparsity)
            
        # Last layer is pruned at half the rate of other layers
        last_layer_sparsity: float = (1 + target_sparsity) / 2
        pruning_rule([layers_to_prune[-1]], [mask_layers[-1]], last_layer_sparsity)

def low_magnitude_pruning(
    layers: list[keras.layers.Layer], 
    mask_layers: list[keras.layers.Layer],
    target_sparsity: float) -> list[np.ndarray]:
    """
    Function which performs low magnitude pruning based on a specified target sparsity
    and updates the corresponding mask layers.

    Args:
        layers (list[keras.layers.Layer]): List of layers to act on.
        mask_layers (list[keras.layers.Layer]): List of masked layers to update with pruning.
        pruning_percentage (float): Target sparsity of the model (% of nonzero weights remaining).
    """

    # Skip if we don't actually need to prune
    if target_sparsity == 1:
        return
    elif target_sparsity > 1 or target_sparsity < 0:
        raise ValueError(f'Found target sparsity of {target_sparsity} but must be between 0 and 1')

    # Calculate the number of weights to prune for each layer
    total_params: int = sum(np.prod(layer.get_weights()[0].shape) + np.prod(layer.get_weights()[1].shape) for layer in layers)
    num_params_to_prune: int = int(total_params * (1 - target_sparsity))

    # Flatten and sort the weights and biases across all layers
    all_weights_biases: list[np.ndarray] = [np.concatenate([layer.get_weights()[0].flatten(), layer.get_weights()[1].flatten()]) for layer in layers]
    all_weights_biases: np.array = np.concatenate(all_weights_biases)
    sorted_weights_biases: np.array = np.sort(np.abs(all_weights_biases))

    # Find the threshold weight value for pruning
    threshold_weight_bias: float = sorted_weights_biases[num_params_to_prune]

    # Apply pruning by setting low magnitude weights to zero
    for layer, mask in zip(layers, mask_layers):
        weights: np.ndarray = layer.get_weights()[0]
        biases: np.ndarray = layer.get_weights()[1]
        
        # Get pruned model weights/biases
        pruned_weights: np.ndarray = np.where(np.abs(weights) < threshold_weight_bias, 0, weights)
        pruned_biases: np.ndarray = np.where(np.abs(biases) < threshold_weight_bias, 0, biases)
        
        # Get new masks after model pruning
        weights_mask: np.ndarray = np.where(np.abs(weights) < threshold_weight_bias, 0, 1)
        biases_mask: np.ndarray = np.where(np.abs(biases) < threshold_weight_bias, 0, 1)
        
        # Update pruned layer weights and masks
        layer.set_weights([pruned_weights, pruned_biases])
        mask.set_weights([weights_mask, biases_mask])