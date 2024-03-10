"""
pruning.py

Code for performing model pruning.
"""

import numpy as np
import tensorflow as tf

from src.model import LeNet, load_model, save_model

def iterative_magnitude_pruning(feature_shape: tuple[int, ...], num_classes: int, model_index: int, 
X_train: np.array, Y_train: np.array, X_test: np.array, Y_test: np.array, 
epochs_per_step: int, total_pruning_percentage: float, pruning_steps: int):
    """
    Function to perform iterative magnitude pruning on the model and save each pruning step's model parameters.

    :param feature_shape:            Shape of input feature instances.
    :param num_classes:              Number of output labels.
    :param model_index:              Model index to perform iterative magnitude pruning on.
    :param X_train:                  Training instances.
    :param X_test:                   Testing instances.
    :param Y_train:                  Training labels.
    :param Y_test:                   Testing labels.
    :param epochs_per_step:          Number of epochs to train each model for in each step.
    :param total_pruning_percentage: Total amount of pruning to do as a percentage from 0 - 1.
    :param pruning_steps:            Number of steps to perform pruning over.
    """

    # Load the trained model at pruning step 0
    model, callbacks = load_model(feature_shape, num_classes, model_index, 0, True)
    per_step_pruning: float = total_pruning_percentage / pruning_steps

    # Iterate over each pruning step
    for step in range(pruning_steps):

        # Prune the model
        pruned_model = prune_model(model, per_step_pruning)

        # Train the pruned model
        pruned_model.fit(X_train, Y_train, epochs=epochs_per_step, validation_data=(X_test, Y_test), callbacks=callbacks, verbose=1)

        # Replace the model for next iterator
        model = pruned_model

        # Save the pruned model with the corresponding pruning percentage
        save_model(model, model_index, step + 1, True)

        # TODO: Reset non-pruned weights to original values
        original_model, callbacks = load_model(feature_shape, num_classes, model_index, 0, False)


def prune_model(model: LeNet, pruning_percentage: float) -> LeNet:
    """
    Function to prune a given model according to the specified percentage.

    :param model:                 Model to be pruned.
    :param pruning_percentage:    Percentage of weights to prune.

    :returns: Pruned model.
    """
    assert 0 <= pruning_percentage < 1, 'Pruning % must be between 0 and 1'
    print(f'Pruning Percentage: {pruning_percentage}')

    # Get all the layers which we can prune
    prunable_layers = [layer for layer in model.layers 
                        if isinstance(layer, tf.keras.layers.Conv2D) 
                        or isinstance(layer, tf.keras.layers.Dense)]

    # Compute the global threshold to prune based on pruning percentage

    # Perform layer-wise pruning
    for layer in prunable_layers:
        # prune_layer(layer, global_threshold)
        pass

    # Compile the pruned model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def prune_layer(layer: tf.keras.layers.Layer, threshold: float):
    """
    Function to prune a layer using magnitude-based pruning.

    :param layer:     Model layer to prune.
    :param threshold: Float value to prune weights at or below.
    """

    weights = layer.get_weights()
    # Get all weights below a magnitude threshold and set them to 0 then update the layer mask
    pruned_weights = [np.where(np.abs(w) < threshold, 0, w) for w in weights]
    mask = [np.where(np.abs(w) < threshold, 0, 1) for w in weights]  # Create mask
    layer.set_weights(pruned_weights)
    layer.set_mask(mask)
