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

from src.harness.model import LeNet300, save_model

def train(make_dataset: callable, model: LeNet300, optimizer: tf.keras.optimizers.Optimizer, iterations: int):
    """
    Function to perform training for a model.

    :param make_dataset:     Function to produce the training/test sets.
    :param model:            Model to optimize.
    :param optimizer:        Optimizer to use during training.
    :param iterations:       Number of iterations to train for.

    :returns: A dictionary of the weights before and after training.
    """

    # Save network initial weights and masks
    initial_weights: dict[str: np.array] = model.get_current_weights()
    save_model(model, 0, untrained=True)

    def training_loop(model: LeNet300):
        """
        Main training loop for the model.
        """
        iteration: int = 0
        # Extract input and target
        inputs, labels, _, _ = make_dataset()
        
        while iteration < iterations:
            # Forward pass
            with tf.GradientTape() as tape:
                # Create the new model using the previous model's weights/masks
                model = LeNet300(model.seed, inputs, labels, presets=model.get_current_weights(), masks=model.masks)

                # Compute gradients
                gradients = tape.gradient(model.loss, model.weights.values())

                # Update weights
                optimizer.apply_gradients(zip(gradients, model.weights.values()))

            iteration += 1

            # Print training progress
            print(f"Iteration {iteration}/{iterations}, Loss: {model.loss.numpy()}")

            # Save model weights after each iteration
            save_model(model, iteration, False)

    # Run the training loop
    training_loop(model)

    # Save network final weights and masks
    save_model(model, 0, final=True)

    # Return initial and final weights
    return initial_weights, model.get_current_weights()