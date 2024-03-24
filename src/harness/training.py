# Copyright (C) 2018 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
training.py

Module containing function to train a neural network.

Modified By: Jordan Bourdeau
Date: 3/17/24
"""

import numpy as np
import tensorflow as tf

from src.harness.constants import Constants as C
from src.harness.model import LeNet300, save_model


class TrainingRound:
    def __init__(self, 
                 pruning_step: int,
                 # Model parameters
                 initial_weights: list[np.ndarray],
                 final_weights: list[np.ndarray],
                 masks: list[np.ndarray],
                 # Metrics
                 train_losses: np.array,
                 train_accuracies: np.array,
                 test_losses: np.array,
                 test_accuracies: np.array,
                 ):
        """
        Class containing data from a single 

        Parameters:
        :param pruning_step:                 (int) Integer for the step in pruning. 
        :param initial_weights:              (list[np.ndarray]) Initial weights of the model.
        :param final_weights:                (list[np.ndarray]) Final weights of the model.
        :param masks:                        (list[np.ndarray]) List of mask model weights (binary mask).

        ...
        """
        self.pruning_step: int = pruning_step
        self.initial_weights: list[np.ndarray] = initial_weights
        self.final_weights: list[np.ndarray] = final_weights
        self.masks: list[np.ndarray] = masks

        self.train_losses: np.array = train_losses
        self.train_accuracies: np.array = train_accuracies
        self.test_losses: np.array = test_losses
        self.test_accuracies: np.array = test_accuracies


# def train(make_dataset: callable, 
#           model: LeNet300, 
#           pruning_step: int = 0,
#           optimizer: tf.keras.optimizers.Optimizer = C.OPTIMIZER(), 
#           iterations: int = C.TRAINING_ITERATIONS):
#     """
#     Function to perform training for a model.

#     :param make_dataset:     Function to produce the training/test sets.
#     :param model:            Model to optimize.
#     :param pruning_step:     Integer value for the step in pruning. Defaults to 0.
#     :param optimizer:        Optimizer to use during training.
#     :param iterations:       Number of iterations to train for.

#     :returns: A dictionary of the weights before and after training.
#     """

#     # Save network initial weights and masks
#     initial_weights: dict[str: np.ndarray] = model.get_current_weights()
#     initial_masks: dict[str: np.array] = model.masks
#     untrained_accuracy: float = model.accuracy
#     untrained_loss: float = model.loss

#     if pruning_step == 0:
#         save_model(model, pruning_step, untrained=True)

#     def training_loop():
#         """
#         Main training loop for the model.
#         """
#         nonlocal model
#         nonlocal make_dataset

#         iteration: int = 0
#         # Extract input and target
#         training_inputs, training_labels, _, _ = make_dataset()
        
#         while iteration < iterations:
#             model.training_step(training_inputs, training_labels, optimizer)
#             iteration += 1
#             # Print training progress
#             print(f"Iteration {iteration}/{iterations}, Loss: {model.loss.numpy()}")

#     # Run the training loop
#     training_loop()

#     # Save network final weights and masks to its folder in the appropriate trial/final folder
#     save_model(model, pruning_step)
#     final_weights: dict[str: np.ndarray] = model.get_current_weights()
#     final_masks: dict[str: np.array] = model.masks
#     trained_training_accuracy: float = model.accuracy
#     trained_training_loss: float = model.loss

#     # Evaluate model on test set
#     _, _, testing_inputs, testing_labels, = make_dataset()
#     model = LeNet300(model.seed, testing_inputs, testing_labels, presets=final_weights, masks=model.masks)
#     trained_test_accuracy: float = model.accuracy
#     trained_test_loss: float = model.loss

#     # Save training round data and return it
#     round: TrainingRound = TrainingRound(
#         pruning_step, 
#         initial_weights, final_weights, 
#         initial_masks, final_masks, 
#         untrained_accuracy, untrained_loss, 
#         trained_training_accuracy, trained_training_loss,
#         trained_test_accuracy, trained_test_loss
#     )

#     return round