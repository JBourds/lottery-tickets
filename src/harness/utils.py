"""
utils.py

File containing utility functions.
"""

import os
import tensorflow as tf
import tensorflow_model_optimization as tfmot

import src.harness.constants as C

# Aliases
ConstantSparsity = tfmot.sparsity.keras.ConstantSparsity
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
UpdatePruningStep = tfmot.sparsity.keras.UpdatePruningStep
Callback = tf.keras.callbacks.Callback
TensorBoard = tf.keras.callbacks.TensorBoard
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint


# def get_model_callbacks(model_index: int, pruning_step: int) -> list[Callback]:
#     """
#     Function to return all associated callbacks with a model.

#     :param model_index:  Integer index for the model.
#     :param pruning_step: Step of model pruning

#     :returns: List of callbacks to use when fitting the model.
#     """
#     # Create the callbacks
#     model_name: str = get_model_name(model_index, pruning_step)
#     tensorboard_path: str = get_model_directory(model_index, C.FIT_DIRECTORY)
#     checkpoint_path: str = get_model_directory(model_index, C.CHECKPOINT_DIRECTORY)

#     # Create the model checkpoint callback
#     callbacks: list[Callback] = [
#         TensorBoard(log_dir=tensorboard_path, histogram_freq=1),
#         ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True),
#         UpdatePruningStep(),
#     ]

#     return callbacks

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