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
from tensorflow_model_optimization.sparsity import keras as sparsity

from src.harness.constants import Constants as C
from src.harness.metrics import get_train_test_loss_accuracy
from src.harness.model import save_model
from src.harness.utils import count_nonzero_parameters


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

@tf.function
def train_one_step(model: tf.keras.Model,
                   mask_model: tf.keras.Model, 
                   inputs: tf.Tensor, 
                   labels: tf.Tensor,
                   train_loss: tf.keras.metrics.Metric,
                   train_accuracy: tf.keras.metrics.Metric,
                   loss_fn: tf.keras.losses.Loss = tf.keras.losses.CategoricalCrossentropy(), 
                   optimizer: tf.keras.optimizers.Optimizer = C.OPTIMIZER(), 
                   ):
    """
    Function to compute one step of gradient descent optimization
    """

    with tf.GradientTape() as tape:
        # Make predictions using defined model-
        y_pred = model(inputs)

        # Compute loss
        loss = loss_fn(labels, y_pred)

    # Compute gradients with respect to defined loss and weights and biases
    gradients = tape.gradient(loss, model.trainable_variables)

    # List to hold element-wise multiplication between
    # computed gradient and masks
    grad_mask_mul = []

    # Perform element-wise multiplication between computed gradients and masks
    for grad_layer, mask in zip(gradients, mask_model.trainable_weights):
        grad_mask_mul.append(tf.math.multiply(grad_layer, mask))

    # Apply computed gradients to model's weights and biases
    optimizer.apply_gradients(zip(grad_mask_mul, model.trainable_variables))

    # Compute accuracy
    return train_loss(loss), train_accuracy(labels, y_pred)

@tf.function
def test_step(model: tf.keras.Model, 
              data: tf.Tensor, 
              labels: tf.Tensor,
              test_loss: tf.keras.metrics.Metric,
              test_accuracy: tf.keras.metrics.Metric,
              loss_fn: tf.keras.losses.Loss = tf.keras.losses.CategoricalCrossentropy(),
              ):
    """
    Function to test model performance on testing dataset
    """
    predictions = model(data)
    loss = loss_fn(labels, predictions)
    return test_loss(loss), test_accuracy(predictions, labels)

# Training function
def training_loop(
        pruning_step: int,
        model_stripped: tf.keras.Model, 
        mask_model_stripped: tf.keras.Model,
        make_dataset: callable,
        make_train_test_loss_accuracy: callable,
        num_epochs: int, 
        patience: int,
        minimum_delta: float,
        optimizer: tf.keras.optimizers.Optimizer = C.OPTIMIZER(),
    ) -> tf.keras.Model:
    """
    Main training loop for the model.
    """
    # Number of epochs without improvement
    local_patience: int = 0
    best_test_loss: float = float('inf')

    # Create metrics
    train_loss, train_accuracy, test_loss, test_accuracy = get_train_test_loss_accuracy()

    # Extract input and target
    X_train, Y_train, X_test, Y_test = make_dataset()

    initial_weights: list[np.ndarray] = model_stripped.trainable_weights
    masks: list[np.ndarray] = mask_model_stripped.trainable_weights
    train_losses: np.array = np.zeros(Y_train.shape[0])
    train_accuracies: np.array = np.zeros(Y_train.shape[0])
    test_losses: np.array = np.zeros(Y_test.shape[0])
    test_accuracies: np.array = np.zeros(Y_test.shape[0])

    for epoch in range(num_epochs):
        
        # Exit early if there are `patience` epochs without improvement
        if local_patience >= patience:
            print(f'Early stopping initiated')
            break
        
        # Update model parameters for each point in the training set
        for x, y in zip(X_train, Y_train):
            train_loss, train_accuracy = train_one_step(
                model_stripped, 
                mask_model_stripped, 
                x, 
                y, 
                train_loss,
                train_accuracy,
                optimizer=optimizer
            )
            train_losses[idx] = train_loss
            train_accuracies[idx] = train_accuracy

        # Evaluate model on each point in the test set
        for idx, (x_t, y_t) in enumerate(zip(X_test, Y_test)):
            test_loss, test_accuracy = test_step(
                model_stripped, 
                optimizer, 
                x_t, 
                y_t,
                test_loss,
                test_accuracy,
            )
            test_losses[idx] = test_loss
            test_accuracies[idx] = test_accuracy

        # Display output
        print(f'Epoch {epoch + 1}, Train/Test Loss: {train_loss:.4f}/{test_loss:.4f}, Train/Test Accuracy: {train_accuracy:.4f}/{test_accuracy:.4f}')
        print(f'Total number of trainable parameters = {np.sum(count_nonzero_parameters(model_stripped))}')

        # Check for early stopping criteria
        mean_test_loss: float = np.mean(test_losses)
        if mean_test_loss < best_test_loss and (best_test_loss - mean_test_loss) >= minimum_delta:
            # update 'best_test_loss' variable to lowest loss encountered so far
            best_test_loss = mean_test_loss
            # Reset the counter
            local_patience = 0
        else:  # there is no improvement in monitored metric 'val_loss'
            local_patience += 1  # number of epochs without any improvement

    final_weights: list[np.ndarray] = model_stripped.trainable_weights

    # Compile training round data
    round_data: TrainingRound = TrainingRound(pruning_step, initial_weights, final_weights, masks, train_losses, train_accuracies, test_losses, test_accuracies)

    return model_stripped, round_data
    

def train(
    random_seed: int,
    pruning_step: int,
    model: tf.keras.Model, 
    mask_model: tf.keras.Model,
    make_dataset: callable, 
    make_train_test_loss_accuracy: callable,
    num_epochs: int = C.TRAINING_EPOCHS,
    patience: int = C.PATIENCE,
    minimum_delta: float = C.MINIMUM_DELTA,
    optimizer: tf.keras.optimizers.Optimizer = C.OPTIMIZER(), 
    ) -> tuple[tf.keras.Model, tf.keras.Model, TrainingRound]:
    """
    Function to perform training for a model.

    :param pruning_step:     Integer value for the step in pruning. Defaults to 0.
    :param model:            Model to optimize.
    :param mask_model:       Model whose weights correspond to masks being applied.

    :param make_dataset:     Function to produce the training/test sets.
    :param num_epochs:       Number of epochs to train for.
    :param patience:         Number of epochs which can be ran without improvement before calling early stopping.
    :param minimum_delta:    Minimum increase to be considered an improvement.s
    :param optimizer:        Optimizer to use during training.

    :returns: Model, masked model, and training round objects with the final trained model and the training summary/.
    """


    # Run the training loop
    model, training_round = training_loop(
        pruning_step, 
        sparsity.strip_pruning(model), 
        sparsity.strip_pruning(mask_model), 
        make_dataset, 
        num_epochs, 
        patience, 
        minimum_delta, 
        optimizer,
    )

    # Save network final weights and masks to its folder in the appropriate trial folder
    save_model(model, random_seed, pruning_step)
    save_model(mask_model, random_seed, pruning_step, masks=True)

    return model, mask_model, training_round