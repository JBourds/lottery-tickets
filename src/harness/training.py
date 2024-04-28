"""
training.py

Module containing functions to train a neural network.
This is necessary since we must use gradient type to mask updates to
the neural network.

Modified By: Jordan Bourdeau
Date: 3/17/24
"""

import numpy as np
import os
import tensorflow as tf

from src.harness import constants as C
from src.harness import dataset as ds
from src.harness import history
from src.harness import model as mod
from src.harness import paths as paths
from src.harness import utils

def get_train_one_step() -> callable:
    
    @tf.function
    def train_one_step(
        model: tf.keras.Model, 
        mask_model: tf.keras.Model, 
        inputs: tf.Tensor, 
        labels: tf.Tensor, 
        optimizer: tf.keras.optimizers.Optimizer,
        loss_fn: tf.keras.metrics.Metric = C.LOSS_FUNCTION(),
        accuracy_metric: tf.keras.metrics.Metric = tf.keras.metrics.CategoricalAccuracy(),
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
        :param accuracy_metric:   Accuracy metric to be used. Defaults to value in `constants.py`.

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

        accuracy: float = accuracy_metric(labels, predictions)
        accuracy_metric.reset_state()
        
        return loss, accuracy
    
    return train_one_step
    
@tf.function(experimental_relax_shapes=True)
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
        
    predictions: tf.Tensor = model(inputs)
    loss: float = loss_fn(labels, predictions)
    accuracy: float = accuracy_metric(labels, predictions)
    accuracy_metric.reset_state()
    return loss, accuracy

def training_loop(
    pruning_step: int,
    model: tf.keras.Model, 
    mask_model: tf.keras.Model,
    dataset: ds.Dataset,
    num_epochs: int = C.TRAINING_EPOCHS, 
    batch_size: int = C.BATCH_SIZE,
    patience: int = C.PATIENCE,
    minimum_delta: float = C.MINIMUM_DELTA,
    loss_fn: tf.keras.losses.Loss = C.LOSS_FUNCTION(),
    optimizer: tf.keras.optimizers.Optimizer = C.OPTIMIZER(),
    allow_early_stopping: bool = True,
    verbose: bool = True,
    ) -> tuple[tf.keras.Model, history.TrialData]:
    """
    Main training loop for the model.

    :param pruning_step:  Integer for the # pruning step the model is on. Used for saving model weights.
    :param model:         Keras model with weights being trained for performance.  
    :param mask_model:    Keras model whose weights signify the masks to use on gradient updates. 
    :param dataset:       Python enum for the dataset being used.
    :param num_epochs:    Integer value for the number of epochs to run. Has a default value in `constants.py`.
    :param batch_size:    Size of batches to train on. Has a default value in `constants.py`.
    :param patience:      Number of rounds before a model is considered to no longer be improving for early stopping. 
                          Has a default value in `constants.py`.
    :param minimum_delta: Minimum accuracy improvement to be considered an improvement.
                          Has a default value in `constants.py`.
    :param loss_fn:       Loss function being used. Has a default value in `constants.py`.
    :param optimizer:     Optimizer function being used to update model weights. Has a default value in `constants.py`.
    :param verbose:       Whether console output is emitted or not.

    :returns: Model with updated weights as well as the training round data.
    """
    # Number of epochs without improvement
    local_patience: int = 0
    best_test_loss: float = float('inf')

    # Extract input and target
    X_train, X_test, Y_train, Y_test = dataset.load()

    initial_parameters: list[np.ndarray] = [np.copy(weight) for weight in model.get_weights()]
    masks: list[np.ndarray] = [np.copy(weight) for weight in mask_model.get_weights()]

    # Store the loss and accuracies at various points to use later in history.TrialData object
    train_losses: np.array = np.zeros(num_epochs)
    train_accuracies: np.array = np.zeros(num_epochs)
    test_losses: np.array = np.zeros(num_epochs)
    test_accuracies: np.array = np.zeros(num_epochs)

    # Calculate the number of batches and create arrays to keep track of batch
    # loss/accuracies while iterating over batches before it goes into training loss/accuracies
    num_batches: int = int(np.ceil(len(X_train) / batch_size))
    batch_losses: np.array = np.zeros(num_batches)
    batch_accuracies: np.array = np.zeros(num_batches)

    train_one_step: callable = get_train_one_step()
    accuracy_metric: tf.keras.metrics.Metric = tf.keras.metrics.CategoricalAccuracy()

    if verbose:
        print(f'Step {pruning_step} of Iterative Magnitude Pruning')
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
                accuracy_metric,
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
            loss_fn,
            accuracy_metric,
        )
        
        if verbose:
            print(f'Epoch {epoch + 1} Train Loss: {train_losses[epoch]:.3f}, Train Accuracy: {train_accuracies[epoch]:.3f}, Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.3f}')

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
                if verbose:
                    print(f'Early stopping initiated')
                break

    # Compile training round data
    trial_data: history.TrialData = history.TrialData(
        pruning_step, 
        initial_parameters, 
        [np.copy(weight) for weight in model.get_weights()], 
        masks, 
        train_losses, 
        train_accuracies, 
        test_losses, 
        test_accuracies
    )

    return trial_data
    

def train(
    random_seed: int,
    pruning_step: int,
    model: tf.keras.Model, 
    mask_model: tf.keras.Model,
    dataset: ds.Dataset, 
    num_epochs: int = C.TRAINING_EPOCHS,
    batch_size: int = C.BATCH_SIZE,
    patience: int = C.PATIENCE,
    minimum_delta: float = C.MINIMUM_DELTA,
    loss_function: tf.keras.losses.Loss = C.LOSS_FUNCTION(),
    optimizer: tf.keras.optimizers.Optimizer = C.OPTIMIZER(), 
    allow_early_stopping: bool = True,
    ) -> tuple[tf.keras.Model, tf.keras.Model, history.TrialData]:
    """
    Function to perform a single round of training for a model.
    NOTE: Modifed `model` input's weights.

    :param random_seed:   Random seed being used.
    :param pruning_step:  Integer value for the step in pruning.
    :param model:         Model to optimize.
    :param mask_model:    Model whose weights correspond to masks being applied.
    :param dataset:       Python enum for the dataset being used.
    :param num_epochs:    Number of epochs to train for. Has a default value in `constants.py`.
    :param batch_size:    Size of the batches to use during training. Has a default value in `constants.py`.
    :param patience:      Number of epochs which can be ran without improvement before calling early stopping. Has a default value in `constants.py`.
    :param minimum_delta: Minimum increase to be considered an improvement. Has a default value in `constants.py`.
    :param loss_function: Loss function to use during training. Has a default value in `constants.py`.
    :param optimizer:     Optimizer to use during training. Has a default value in `constants.py`.
    :param allow_early_stopping: Boolean flag for whether early stopping is enabled.

    :returns: Model, masked model, and training round objects with the final trained model and the training summary/.
    """

    utils.set_seed(random_seed)

    # Run the training loop
    trial_data = training_loop(
        pruning_step, 
        model, 
        mask_model, 
        dataset, 
        num_epochs, 
        batch_size,
        patience, 
        minimum_delta, 
        loss_function,
        optimizer,
        allow_early_stopping,
    )
    
    # Save the round data
    trial_data_directory: str = paths.get_model_directory(random_seed, pruning_step, trial_data=True)
    trial_data.save_to(trial_data_directory, 'trial_data.pkl')

    # Save network final weights and masks to its folder in the appropriate trial folder
    mod.save_model(model, random_seed, pruning_step)
    mod.save_model(mask_model, random_seed, pruning_step, masks=True)

    return trial_data