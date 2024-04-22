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
from src.harness.utils import count_params


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
def train_one_step(
    model: tf.keras.Model, 
    mask_model: tf.keras.Model, 
    inputs: tf.Tensor, 
    labels: tf.Tensor, 
    optimizer: tf.keras.optimizers.Optimizer,
    loss_fn: tf.keras.metrics.Metric = C.LOSS_FUNCTION(),
    accuracy: tf.keras.metrics.Metric = tf.keras.metrics.CategoricalAccuracy(),
    ):
    """
    Tensorflow function to performa a single step of gradient descent.

    :param model:      Keras model being trained.
    :param mask_model: Model which matches the model being trained but stores 1s and 0s for
                       the mask being applied to the model getting trained.
    :param inputs:     Batch inputs.
    :param labels:     Batch labels.
    :param optimizer:  Optimizer function being used. 
    :param loss_fn:    Loss function being used. Defaults to value in `constants.py`.
    :param accuracy:   Accuracy metric to be used. Defaults to value in `constants.py`.

    :returns: Loss and accuracy from data passed into the model.
    """
    with tf.GradientTape() as tape:
        predictions: tf.Tensor = model(inputs)
        loss: tf.keras.losses.Loss = loss_fn(labels, predictions)
        
    gradients: tf.Tensor = tape.gradient(loss, model.trainable_variables)

    grad_mask_mul: list[tf.Tensor] = []

    for grad_layer, mask in zip(gradients, mask_model.trainable_weights):
        grad_mask_mul.append(tf.math.multiply(grad_layer, mask))

    optimizer.apply_gradients(zip(grad_mask_mul, model.trainable_variables))

    return loss, accuracy(labels, predictions)

@tf.function(experimental_relax_shapes=True)
def test_step(
    model: tf.keras.Model, 
    inputs: tf.Tensor, 
    labels: tf.Tensor,
    loss_fn: tf.keras.metrics.Metric = C.LOSS_FUNCTION(),
    accuracy: tf.keras.metrics.Metric = tf.keras.metrics.CategoricalAccuracy(),
    ) -> tuple[float, float]:
    """
    Function to test model performance on testing dataset.

    :param model:      Keras model being tested.
    :param inputs:     Testing inputs.
    :param labels:     Testing labels.
    :param loss_fn:    Loss function being used. Defaults to value in `constants.py`.
    :param accuracy:   Accuracy metric to be used. Defaults to value in `constants.py`.

    :returns: Loss and accuracy from data passed into the model.
    """
    predictions: tf.Tensor = model(inputs)
    return loss_fn(labels, predictions), accuracy(labels, predictions)

# Training function
def training_loop(
        pruning_step: int,
        model: tf.keras.Model, 
        mask_model: tf.keras.Model,
        make_dataset: callable,
        num_epochs: int = C.TRAINING_EPOCHS, 
        batch_size: int = C.BATCH_SIZE,
        patience: int = C.PATIENCE,
        minimum_delta: float = C.MINIMUM_DELTA,
        loss_fn: tf.keras.losses.Loss = C.LOSS_FUNCTION(),
        optimizer: tf.keras.optimizers.Optimizer = C.OPTIMIZER(),
        allow_early_stopping: bool = True,
    ) -> tuple[tf.keras.Model, TrainingRound]:
    """
    Main training loop for the model.

    :param pruning_step:  Integer for the # pruning step the model is on. Used for saving model weights.
    :param model:         Keras model with weights being trained for performance.  
    :param mask_model:    Keras model whose weights signify the masks to use on gradient updates. 
    :param make_dataset:  Function which produces the inputs/labels.
    :param num_epochs:    Integer value for the number of epochs to run. Has a default value in `constants.py`.
    :param batch_size:    Size of batches to train on. Has a default value in `constants.py`.
    :param patience:      Number of rounds before a model is considered to no longer be improving for early stopping. 
                          Has a default value in `constants.py`.
    :param minimum_delta: Minimum accuracy improvement to be considered an improvement.
                          Has a default value in `constants.py`.
    :param loss_fn:       Loss function being used. Has a default value in `constants.py`.
    :param optimizer:     Optimizer function being used to update model weights. Has a default value in `constants.py`.

    :returns: Model with updated weights as well as the training round data.
    """
    # Number of epochs without improvement
    local_patience: int = 0
    best_test_loss: float = float('inf')

    # Extract input and target
    X_train, X_test, Y_train, Y_test = make_dataset()

    initial_parameters: list[np.ndarray] = model.trainable_variables
    masks: list[np.ndarray] = mask_model.trainable_variables

    # Store the loss and accuracies at various points to use later in TrainingRound object
    train_losses: np.array = np.zeros(num_epochs)
    train_accuracies: np.array = np.zeros(num_epochs)
    test_losses: np.array = np.zeros(num_epochs)
    test_accuracies: np.array = np.zeros(num_epochs)

    # Calculate the number of batches and create arrays to keep track of batch
    # loss/accuracies while iterating over batches before it goes into training loss/accuracies
    num_batches: int = int(np.ceil(len(X_train) / batch_size))
    batch_losses: np.array = np.zeros(num_batches)
    batch_accuracies: np.array = np.zeros(num_batches)

    for epoch in range(num_epochs):
        for batch_index in range(num_batches):
            # Calculate the lower/upper index for batch (assume data is shuffled)
            low_index: int = batch_index * batch_size
            high_index: int = (batch_index + 1) * batch_size
            # Extract data to use for the batch
            X_batch: np.ndarray = X_train[low_index:high_index]
            Y_batch: np.ndarray = Y_train[low_index:high_index]

            # Update model parameters for each point in the training set
            loss, accuracy = train_one_step(
                model, 
                mask_model, 
                X_batch, 
                Y_batch, 
                optimizer,
                loss_fn,
            )

            # Keep track of all the losses/accuracies within the epoch's batches here
            batch_losses[batch_index] = loss
            batch_accuracies[batch_index] = accuracy

        # Set the overall epoch training loss/accuracy to mean of batches
        train_losses[epoch] = np.mean(batch_losses)
        train_accuracies[epoch] = np.mean(batch_accuracies)

        # Evaluate model on the test set using whole batch
        test_loss, test_accuracy = test_step(
            model, 
            X_test, 
            Y_test,
        )

        test_losses[epoch] = test_loss
        test_accuracies[epoch] = test_accuracy

        if allow_early_stopping:
            # Check for early stopping criteria using mean validation loss
            mean_test_loss: float = np.mean(test_losses[:epoch + 1])
            if mean_test_loss < best_test_loss and (best_test_loss - mean_test_loss) >= minimum_delta:
                # update 'best_test_loss' variable to lowest loss encountered so far
                best_test_loss = mean_test_loss
                # Reset the counter
                local_patience = 0
            else:  # there is no improvement in monitored metric 'val_loss'
                local_patience += 1  # number of epochs without any improvement

            # Exit early if there are `patience` epochs without improvement
            if local_patience >= patience:
                print(f'Early stopping initiated')
                break

    final_parameters: list[np.ndarray] = model.trainable_variables

    # Compile training round data
    round_data: TrainingRound = TrainingRound(pruning_step, initial_parameters, final_parameters, masks, train_losses, train_accuracies, test_losses, test_accuracies)

    return model, round_data
    

def train(
    random_seed: int,
    pruning_step: int,
    model: tf.keras.Model, 
    mask_model: tf.keras.Model,
    make_dataset: callable, 
    num_epochs: int = C.TRAINING_EPOCHS,
    batch_size: int = C.BATCH_SIZE,
    patience: int = C.PATIENCE,
    minimum_delta: float = C.MINIMUM_DELTA,
    loss_fn: tf.keras.losses.Loss = C.LOSS_FUNCTION(),
    optimizer: tf.keras.optimizers.Optimizer = C.OPTIMIZER(), 
    allow_early_stopping: bool = True,
    ) -> tuple[tf.keras.Model, tf.keras.Model, TrainingRound]:
    """
    Function to perform training for a model.

    :param random_seed:   Random seed being used.
    :param pruning_step:  Integer value for the step in pruning.
    :param model:         Model to optimize.
    :param mask_model:    Model whose weights correspond to masks being applied.
    :param make_dataset:  Function to produce the training/test sets.
    :param num_epochs:    Number of epochs to train for. Has a default value in `constants.py`.
    :param batch_size:    Size of the batches to use during training. Has a default value in `constants.py`.
    :param patience:      Number of epochs which can be ran without improvement before calling early stopping. Has a default value in `constants.py`.
    :param minimum_delta: Minimum increase to be considered an improvement. Has a default value in `constants.py`.
    :param loss_fn:       Loss function to use during training. Has a default value in `constants.py`.
    :param optimizer:     Optimizer to use during training. Has a default value in `constants.py`.
    :param allow_early_stopping: Boolean flag for whether early stopping is enabled.

    :returns: Model, masked model, and training round objects with the final trained model and the training summary/.
    """

    # Run the training loop
    model, training_round = training_loop(
        pruning_step, 
        sparsity.strip_pruning(model), 
        sparsity.strip_pruning(mask_model), 
        make_dataset, 
        num_epochs, 
        batch_size,
        patience, 
        minimum_delta, 
        loss_fn,
        optimizer,
        allow_early_stopping,
    )

    # Save network final weights and masks to its folder in the appropriate trial folder
    save_model(model, random_seed, pruning_step)
    save_model(mask_model, random_seed, pruning_step, masks=True)

    return model, mask_model, training_round