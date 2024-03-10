"""
utils.py

File containing utility functions.
"""

import os
import tensorflow as tf
import tensorflow_model_optimization as tfmot

import src.constants as C

# Aliases
ConstantSparsity = tfmot.sparsity.keras.ConstantSparsity
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
UpdatePruningStep = tfmot.sparsity.keras.UpdatePruningStep
TensorBoard = tf.keras.callbacks.TensorBoard
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint

def create_path(path: str):
    """
    Helper function to create a path and all its subdirectories.
    :param path: String containing the target path.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created successfully.")
    else:
        print(f"Directory '{path}' already exists.")

def get_model_directory(model_index: int, base_directory: str = "") -> str:
    """
    Function to return the relative directory where a model would go.

    :param base_directory: Base directory to append model subdirectory to. Defaults to empty string.
    :param model_index: Integer for the index/random seed of the model.

    :returns: Returns expected directory for the model.
    """
    return f'{base_directory}model_{model_index}/'

def get_model_name(model_index: int, pruning_step: int = 0, trained: bool = True) -> str:
    """
    Function to return the expected name for a model based on its index and pruning step.

    :param model_index:   Integer for the index/random seed of the model.
    :param pruning_step:  Integer for the pruning iteration.
    :param pretrained:    Boolean for if the model was trained (pretrained saves initial weights).

    :returns: Returns expected name for the model.
    """
    trained: str = 'trained' if trained else 'untrained'
    return f'{trained}_model_{model_index}_step_{pruning_step}.keras'

def get_model_callbacks(model_index: int, pruning_step: int) -> list[tf.keras.callbacks]:
    """
    Function to return all associated callbacks with a model.

    :param model_index:  Integer index for the model.
    :param pruning_step: Step of model pruning

    :returns: List of callbacks to use when fitting the model.
    """
    # Create the callbacks
    model_name: str = get_model_name(model_index, pruning_step)
    tensorboard_path: str = get_model_directory(model_index, C.FIT_DIRECTORY)
    checkpoint_path: str = get_model_directory(model_index, C.CHECKPOINT_DIRECTORY)

    # Create the model checkpoint callback
    callbacks: list[tf.keras.callbacks] = [
        TensorBoard(log_dir=tensorboard_path, histogram_freq=1),
        ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True),
        UpdatePruningStep(),
    ]

    return callbacks

def compare_pruned_unpruned_weights(unpruned_model: tf.keras.Model, pruned_model: tf.keras.Model):
    """
    Function to check the differnce in the pruned vs. unpruned weights.

    :param unpruned_model: Unpruned version of a model.
    :param pruned_model:   Pruned version of a model.
    """
    unpruned_weights = unpruned_model.get_weights()
    pruned_weights = pruned_model.get_weights()

    print(unpruned_weights)
    print(pruned_weights)

    # Calculate percentage of weights set to 0
    total_params = sum(w.size for w in unpruned_weights)
    pruned_params = sum((w == 0).sum() for w in pruned_weights)
    pruned_percentage = pruned_params / total_params * 100

    print(f"Percentage of weights set to 0 after pruning: {pruned_percentage:.2f}%")