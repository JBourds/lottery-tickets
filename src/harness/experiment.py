"""
experiment.py

Module containing code for actually running the lottery ticket hypothesis experiemnts.

Modified By: Jordan Bourdeau
Date: 3/17/24
"""

import functools
import logging
import os
import warnings
from typing import Callable

import multiprocess as mp
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.harness import constants as C
from src.harness import dataset as ds
from src.harness import history
from src.harness import model as mod
from src.harness import paths, pruning, rewind
from src.harness import training as train
from src.harness import utils
from src.harness.architecture import Hyperparameters


def run_experiments(
    starting_seed: int,
    num_experiments: int,
    experiment_directory: str,
    experiment: Callable,
    get_experiment_parameters: callable,
    max_processes: int = os.cpu_count(),
    log_level: int = logging.INFO,
) -> history.ExperimentSummary:
    """
    Main function which runs experiments with provided configurations.
    Optionally performs multiprocessing parallelism to speed up training.

    Args:
        starting_seed (int): Starting random seed to use.
        num_experiments (int): Number of experiments to run with the
            specified configuration.
        experiment_directory (str): String directory for where to put the experiment results.
        experiment (callable): Function to run a single experiment.
        get_experiment_parameters (callable): Function which takes in the seed and experiment
            directory then produces all the parameters which get unpacked into the function
            responsible for running the experiment.
        max_processes (int): Integer value for the maximum number of processes which are attempted
            to be run in parallel.
        log_level (int): Log level to use.
    Returns:
        history.ExperimentSummary: Object containing information about all trained models.
    """

    # Set CPU affinity to use all available CPU cores
    # and keep each thread scheduled to the same core
    os.environ['OMP_NUM_THREADS'] = str(max_processes)
    os.environ['KMP_BLOCKTIME'] = '1'
    os.environ['KMP_SETTINGS'] = '1'
    os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'

    paths.create_path(experiment_directory)
    experiment_summary: history.ExperimentSummary = history.ExperimentSummary()
    experiment_summary.start_timer()

    def run_single_experiment(experiment_arguments: dict) -> tuple[int, history.ExperimentData]:
        """
        Helper function which unpacks experiment arguments into a call to the function.
        Sets logging for the specific experiment, since each experiment runs in
        its own process.
        """
        logging.basicConfig()
        logging.getLogger().setLevel(log_level)
        experiment_data: history.ExperimentData = experiment(
            **experiment_arguments)
        return experiment_data

    # Prepare arguments for multiprocessing
    random_seeds = list(range(starting_seed, starting_seed + num_experiments))
    partial_get_experiment_parameters = functools.partial(
        get_experiment_parameters, directory=experiment_directory)
    experiment_args = [partial_get_experiment_parameters(
        seed) for seed in random_seeds]

    # Run experiments in parallel and collect the results
    with mp.get_context('spawn').Pool(max_processes) as pool:
        experiment_results: list[history.ExperimentData] = list(
            pool.map(run_single_experiment, experiment_args))

    # Fill the experiment summary object, stop its timer, and save it
    experiment_summary.experiments = {
        seed: experiment_result for seed, experiment_result in zip(random_seeds, experiment_results)}
    experiment_summary.stop_timer()
    experiment_summary.save_to(experiment_directory, 'experiment_summary.pkl')
    return experiment_summary


def run_iterative_pruning_experiment(
    random_seed: int,
    create_model: Callable[[None], keras.models.Model],
    hyperparameters: Hyperparameters,
    dataset: ds.Dataset,
    sparsities: list[float],
    pruning_rule: Callable,
    rewind_rule: Callable,
    global_pruning: bool = False,
    experiment_directory: str = './',
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
        global_pruning (bool, optional): Boolean flag for whether pruning is done globally
            or layerwise. Defaults to False.
        experiment_directory (str): Path to place all experimental data.

    Returns:
        history.ExperimentData: Object containing information about all the training rounds produced in the experiment.
    """
    # Set seed for reproducability
    utils.set_seed(random_seed)

    experiment_data = history.ExperimentData()
    experiment_data.start_timer()

    # Make models and save them
    model = create_model()
    mask_model = mod.create_masked_nn(create_model)

    mod.save_model(model, random_seed, 0, initial=True,
                   directory=experiment_directory)
    mod.save_model(mask_model, random_seed, 0, masks=True,
                   initial=True, directory=experiment_directory)

    for pruning_step, sparsity in enumerate(sparsities):
        # Prune the model to the new sparsity and update the mask model
        pruning.prune(model, mask_model, pruning_rule,
                      sparsity, global_pruning=global_pruning)

        # Reset unpruned weights to original values.
        rewind.rewind_model_weights(model, mask_model, rewind_rule)

        trial_data = train.train(
            random_seed=random_seed,
            pruning_step=pruning_step,
            model=model,
            mask_model=mask_model,
            dataset=dataset,
            hp=hyperparameters,
            output_directory=experiment_directory,
        )

        experiment_data.add_trial(trial_data)

        X_train, _, _, _ = dataset.load()
        iteration_count = np.sum(trial_data.train_accuracies != 0)
        logging.info(f'Took {iteration_count} iterations')
        logging.info(
            f'Ended on epoch {np.ceil(iteration_count * hyperparameters.batch_size / X_train.shape[0])} out of {hyperparameters.epochs}')
        logging.info(
            'Best accuracies:\n'
            + f'Training: {np.max(trial_data.train_accuracies) * 100:.2f}%\n'
            + f'Validation: {np.max(trial_data.validation_accuracies) * 100:.2f}%\n'
            + f'Testing: {np.max(trial_data.test_accuracy) * 100:.2f}%\n'
        )

    experiment_data.stop_timer()
    return experiment_data
