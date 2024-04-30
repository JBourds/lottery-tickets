"""
experiment.py

Module containing code for actually running the lottery ticket hypothesis experiemnts.

Modified By: Jordan Bourdeau
Date: 3/17/24
"""

import functools
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

from src.harness import constants as C
from src.harness import dataset as ds
from src.harness import history
from src.harness import model as mod
from src.harness import paths
from src.harness import pruning 
from src.harness import rewind
from src.harness import training as train
from src.harness import utils
            
def run_experiments(
    starting_seed: int,
    num_experiments: int, 
    experiment_directory: str,
    experiment: callable,
    get_experiment_parameters: callable, 
    ) -> history.ExperimentSummary:
    """
    Function where experimental parameters are configured to run.

    Args:
        starting_seed (int): Starting random seed to use.
        num_experiments (int): Number of experiments to run with the
            specified configuration.
        experiment_directory (str): String directory for where to put the experiment results.
        get_experiment_parameters (callable): Function which takes in the seed and experiment
            directory then produces all the parameters which get unpacked into the function
            responsible for running the experiment.
        experiment (callable): Function to run the experiments.
        verbose (bool): Whether training displays console output.

    Returns:
        history.ExperimentSummary: Object containing information about all trained models.
    """
    
    # Make the path to store all the experiment data in
    paths.create_path(experiment_directory)
    
    # Object to keep track of experiment data
    experiment_summary: history.ExperimentSummary = history.ExperimentSummary()
    # For each experiment, use a different random seed and keep track of all the data produced
    for seed in range(starting_seed, starting_seed + num_experiments):
        kwargs: dict = get_experiment_parameters(seed, experiment_directory)
        experiment_data: history.ExperimentData = experiment(**kwargs)
        experiment_summary.add_experiment(seed, experiment_data)
        
        # Save pickled experiment summary after every iteration- works like a checkpoint
        experiment_summary.save_to(experiment_directory, 'experiment_summary.pkl')
    
    # Return the experiment summary still in memory
    return experiment_summary    

def run_iterative_pruning_experiment(
    random_seed: int, 
    create_model: callable, 
    dataset: ds.Dataset,
    sparsities: list[float], 
    pruning_rule: callable,
    rewind_rule: callable,
    loss_function: callable = None,
    optimizer: tf.keras.optimizers.Optimizer = None, 
    global_pruning: bool = False,
    num_epochs: int = C.TRAINING_EPOCHS,
    batch_size: int = C.BATCH_SIZE,
    patience: int = C.PATIENCE,
    minimum_delta: float = C.MINIMUM_DELTA,
    allow_early_stopping: bool = True,
    experiment_directory: str = './',
    verbose: bool = True,
    ) -> history.ExperimentData:
    """
    Function used to run the pruning experiements for a given random seed.
    Will perform iterative pruning given a specified pruning technique
    (e.g. Magnitude-based pruning) and a given rewind technique
    (e.g. Rewind to initial weights).

    Args:
        random_seed (int): Random seed for reproducability.
        create_model (callable): Function which produces the model.
        dataset (enum): Enum for the dataset being used.
        sparsities (list[float]): List of sparsities for each step of training.
        pruning_rule (callable, optional): Function used to prune model. 
        rewind_rule (callable, optional): Function used for rewinding model weights. 
        loss_function (callable, optional): Loss function for training. 
            Defaults to None - becomes loss function specified in `constants.py`.
        optimizer (tf.keras.optimizers.Optimizer, optional): Optimizer. 
            Defaults to None - becomes optimizer specified in `constants.py`..
        global_pruning (bool, optional): Boolean flag for whether pruning is done globally
            or layerwise. Defaults to False.
        num_epochs (int, optional): Number of epochs to train for. Defaults to C.TRAINING_EPOCHS.
        batch_size (int, optional): Batch size to use. Defaults to C.BATCH_SIZE.
        patience (int, optional): Number of epochs training will continue for without improvement
            before implementing early stopping. Defaults to C.PATIENCE.
        minimum_delta (float, optional): Minimum positive change required to count as an improvement. 
            Defaults to C.MINIMUM_DELTA.
        allow_early_stopping (bool, optional): Boolean flag for whether early stopping is enabled. 
            Defaults to True.
        experiment_director (str): Path to place all experimental data.
        verbose (bool, optional): Boolean flag for whether console output is displayed. Defaults to True.

    Returns:
        history.ExperimentData: Object containing information about all the training rounds produced in the experiment.
    """
    # Set seed for reproducability
    utils.set_seed(random_seed)
        
    experiment_data: history.ExperimentData = history.ExperimentData()
    # Make models and save them
    model: keras.Model = create_model()
    mask_model: keras.Model = mod.create_masked_nn(create_model)   
    
    print(f'Experiment directory: {experiment_directory}')
    
    mod.save_model(model, random_seed, 0, initial=True, directory=experiment_directory)
    mod.save_model(mask_model, random_seed, 0, masks=True, initial=True, directory=experiment_directory)
    
    for pruning_step, sparsity in enumerate(sparsities):
        # Prune the model to the new sparsity and update the mask model
        pruning.prune(model, mask_model, pruning_rule, sparsity, global_pruning=global_pruning)

        # Reset unpruned weights to original values.
        rewind.rewind_model_weights(model, mask_model, rewind_rule)
        
        # Handle default initialization
        loss_fn: tf.losses.Loss = C.LOSS_FUNCTION() if loss_function is None else loss_function()
        opt: tf.optimizers.Optimizer = C.OPTIMIZER() if optimizer is None else optimizer()

        trial_data: train.TrialData = train.train(
            random_seed, 
            pruning_step, 
            model, 
            mask_model, 
            dataset, 
            num_epochs=num_epochs, 
            batch_size=batch_size, 
            patience=patience, 
            minimum_delta=minimum_delta,
            loss_function=loss_fn,
            optimizer=opt,
            allow_early_stopping=allow_early_stopping,
            output_directory=experiment_directory,
            verbose=verbose,
        )

        experiment_data.add_pruning_round(trial_data)

        if verbose:
            X_train, _, _, _ = dataset.load()
            iteration_count: int = np.sum(trial_data.train_accuracies != 0)
            print(f'Took {iteration_count} iterations')
            print(f'Ended on epoch {np.ceil(iteration_count * batch_size / X_train.shape[0])} out of {num_epochs}')
            print(f'Ended with a best training accuracy of {np.max(trial_data.train_accuracies) * 100:.2f}% and test accuracy of {np.max(trial_data.test_accuracies) * 100:.2f}%')
        
    return experiment_data