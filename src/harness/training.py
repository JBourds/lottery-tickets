"""
training.py

Module containing functions to train a neural network.
This is necessary since we must use gradient type to mask updates to
the neural network.

Modified By: Jordan Bourdeau
Date: 3/17/24
"""

import copy
import logging
import os
from datetime import datetime

import numpy as np
import tensorflow as tf

from src.harness import constants as C
from src.harness import dataset as ds
from src.harness import history
from src.harness import model as mod
from src.harness import paths as paths
from src.harness import utils
from src.harness.architecture import Hyperparameters


def get_train_one_step() -> callable:

    @tf.function
    def train_one_step(
        model: tf.keras.Model,
        mask_model: tf.keras.Model,
        inputs: tf.Tensor,
        labels: tf.Tensor,
        optimizer: tf.keras.optimizers.Optimizer,
        loss_fn: tf.keras.metrics.Metric,
        accuracy_metric: tf.keras.metrics.Metric = tf.keras.metrics.CategoricalAccuracy(),
    ):
        """
        Tensorflow function to performa a single step of gradient descent.

        :param model:      Keras model being trained.
        :param mask_model: Model which matches the model being trained but stores 1s and 0s for
                           the mask being applied to the model getting trained.
        :param inputs:     Batch inputs.
        :param labels:     Batch labels.
        :param hp.optimizer:  Optimizer function being used.
        :param loss_fn:    Loss function being used. Defaults to value in `constants.py`.
        :param accuracy_metric:   Accuracy metric to be used. Defaults to value in `constants.py`.

        :returns: Loss and accuracy from data passed into the model.
        """
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = loss_fn(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_weights)

        grad_mask_mul = []

        for grad_layer, mask in zip(gradients, mask_model.trainable_weights):
            grad_mask_mul.append(tf.math.multiply(grad_layer, mask))

        optimizer.apply_gradients(
            zip(grad_mask_mul, model.trainable_weights))

        accuracy = accuracy_metric(labels, predictions)
        accuracy_metric.reset_state()

        return loss, accuracy

    return train_one_step


@tf.function
def test_step(
    model: tf.keras.Model,
    inputs: tf.Tensor,
    labels: tf.Tensor,
    loss_fn: tf.keras.metrics.Metric,
    accuracy_metric: tf.keras.metrics.Metric,
) -> tuple[float, float]:
    """
    Function to test model performance on testing dataset.
    Note: Make sure you clear the loss function and metric between calls!

    :param model:      Keras model being tested.
    :param inputs:     Testing inputs.
    :param labels:     Testing labels.
    :param loss_fn:    Loss function being used. Defaults to value in `constants.py`.
    :param accuracy_metric:   Accuracy metric to be used. Defaults to value in `constants.py`.

    :returns: Loss and accuracy from data passed into the model.
    """

    predictions = model(inputs)
    loss = loss_fn(labels, predictions)
    accuracy = accuracy_metric(labels, predictions)
    accuracy_metric.reset_state()
    return loss, accuracy


def training_loop(
    pruning_step: int,
    model: tf.keras.Model,
    mask_model: tf.keras.Model,
    dataset: ds.Dataset,
    hp: Hyperparameters,
) -> history.TrialData:
    """
    Main training loop for the model.
    :returns Model with updated weights as well as the training round data.
    """
    # Keep track of the start time
    training_start_time = datetime.now()

    # Number of epochs without improvement
    local_patience = 0
    best_validation_loss = float('inf')

    # Extract input and target for training and test/validation set
    X_train, X_test, Y_train, Y_test = dataset.load()

    initial_parameters = [tensor.numpy()
                          for tensor in copy.deepcopy(model.trainable_weights)]
    masks = [tensor.numpy for tensor in copy.deepcopy(
        mask_model.trainable_weights)]

    # Calculate the number of batches and create arrays to keep track of batch
    # loss/accuracies while iterating over batches before it goes into training loss/accuracies
    num_validation_instances = int(len(X_train) * 0.1)
    num_batches = int(
        np.ceil((len(X_train) - num_validation_instances) / hp.batch_size))

    # Store the loss and accuracies at various points to use later in history.TrialData object
    train_losses = np.zeros(hp.epochs * num_batches)
    train_accuracies = np.zeros(hp.epochs * num_batches)
    validation_losses = np.zeros(
        int(hp.epochs * np.ceil(num_batches / hp.eval_freq)))
    validation_accuracies = np.zeros(
        int(hp.epochs * np.ceil(num_batches / hp.eval_freq)))

    # Initialize function and metric for use
    train_one_step = get_train_one_step()

    loss_function = hp.loss_function()
    optimizer = hp.optimizer()
    accuracy_metric = hp.accuracy_metric()

    loss_before_training, accuracy_before_training = test_step(
        model,
        X_test,
        Y_test,
        loss_function,
        accuracy_metric,
    )

    logging.info(f'Number of Batches: {num_batches}')
    logging.info(f'Batch Size: {hp.batch_size}')
    logging.info(f'Step {pruning_step} of Iterative Magnitude Pruning')

    batch_counter = 0

    for epoch in range(hp.epochs):
        # Shuffle training data using pruning step as a seed
        tf.random.shuffle(X_train, seed=pruning_step)
        tf.random.shuffle(Y_train, seed=pruning_step)

        for batch_index in range(num_batches):
            # Calculate the lower/upper index for batch (assume data is shuffled)
            low_index = batch_index * hp.batch_size
            high_index = (batch_index + 1) * hp.batch_size
            # Extract data to use for the batch
            X_batch = X_train[low_index:high_index]
            Y_batch = Y_train[low_index:high_index]

            # Update model parameters for each point in the training set
            loss, accuracy = train_one_step(
                model,
                mask_model,
                X_batch,
                Y_batch,
                optimizer,
                loss_function,
                accuracy_metric,
            )

            # Keep track of all the losses/accuracies within the epoch's batches here
            train_losses[epoch * num_batches + batch_index] = loss
            train_accuracies[epoch * num_batches + batch_index] = accuracy

            # Only perform checks for early stopping at specified number of intervals
            if batch_counter % hp.eval_freq == 0:
                # Evaluate model on validation test set using whole batch after each epoch
                validation_loss, validation_accuracy = test_step(
                    model,
                    X_test,
                    Y_test,
                    loss_function,
                    accuracy_metric,
                )

                logging.info(
                    f'Epoch {epoch + 1}, Batch {batch_index} '
                    + f'Validation Loss: {validation_loss}, '
                    + f'Validation Accuracy: {validation_accuracy}'
                )

                validation_index = batch_counter // hp.eval_freq
                validation_losses[validation_index] = validation_loss
                validation_accuracies[validation_index] = validation_accuracy

                if hp.early_stopping:
                    # Check for early stopping criteria using mean validation loss
                    mean_validation_loss = np.mean(
                        validation_losses[:validation_index + 1])
                    if mean_validation_loss < best_validation_loss and (best_validation_loss - mean_validation_loss) >= hp.minimum_delta:
                        # update 'best_validation_loss' variable to lowest loss encountered so far
                        best_validation_loss = mean_validation_loss
                        local_patience = 0
                    else:  # there is no improvement in monitored metric 'val_loss'
                        local_patience += 1
                    # Exit early if there are `hp.patience` successive validations without improvement
                    if local_patience >= hp.patience:
                        logging.info(f'Early stopping initiated')
                        break

            logging.debug(
                f'Epoch {epoch + 1} Iteration {batch_index + 1} with Batch Size: {hp.batch_size}: Training Loss: {loss}, Training Accuracy: {accuracy}')

            batch_counter += 1

        # Cursed way to ripple break from inner loop to outer loop
        else:
            continue
        break

    best_loss = np.min(validation_losses[validation_losses != 0])
    logging.info(
        f'Best Validation Accuracy and Loss: {np.max(validation_accuracies)}, {best_loss}')

    final_parameters = [tensor.numpy()
                        for tensor in copy.deepcopy(model.trainable_weights)]

    # Compile training round data
    trial_data = history.TrialData(
        pruning_step,
        initial_parameters,
        final_parameters,
        masks,
        loss_before_training,
        accuracy_before_training,
        train_losses,
        train_accuracies,
        validation_losses,
        validation_accuracies,
    )
    trial_data.set_start_time(training_start_time)

    return trial_data


def train(
    random_seed: int,
    pruning_step: int,
    model: tf.keras.Model,
    mask_model: tf.keras.Model,
    dataset: ds.Dataset,
    hp: Hyperparameters,
    output_directory: str = './',
) -> history.TrialData:
    """
    Function to perform a single round of training for a model.
    NOTE: Modifed `model` input's weights.

    :returns Model, masked model, and training round objects with the final trained model and the training summary.
    """

    utils.set_seed(random_seed)

    # Run the training loop
    trial_data = training_loop(
        pruning_step,
        model,
        mask_model,
        dataset,
        hp,
    )

    # Stop the trial data's timer and save it to the target directory
    trial_data.stop_timer()
    trial_data_directory: str = os.path.join(output_directory, paths.get_model_directory(
        random_seed, pruning_step, trial_data=True))
    trial_data.save_to(trial_data_directory, 'trial_data.pkl')

    # Save network final weights and masks to its folder in the appropriate trial folder
    mod.save_model(model, random_seed, pruning_step,
                   directory=output_directory)
    mod.save_model(mask_model, random_seed, pruning_step,
                   masks=True, directory=output_directory)

    return trial_data
