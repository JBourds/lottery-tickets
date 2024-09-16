"""
experiment.py

Module containing code for actually running the lottery ticket hypothesis experiemnts.

Modified By: Jordan Bourdeau
Date: 3/17/24
"""

import functools
import gc
import logging
import multiprocess as mp
import numpy as np
import os
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow import keras
from typing import Callable, Dict, Tuple

from src.harness import constants as C
from src.harness import dataset as ds
from src.harness import model as mod
from src.harness import paths, pruning, rewind
from src.harness import training as train
from src.harness import utils
from src.harness.architecture import Hyperparameters


def run_experiments(
    starting_seed: int,
    num_experiments: int,
    experiment_directory: str,
    experiment: Callable[[Dict], None],
    get_experiment_parameters: Callable[[int, str], Dict],
    log_level: int = logging.INFO,
):
    """
    Main function which runs experiments with provided configurations.

    Args:
        starting_seed (int): Starting random seed to use.
        num_experiments (int): Number of experiments to run with the
            specified configuration.
        experiment_directory (str): String directory for where to put the experiment results.
        experiment (callable): Function to run a single experiment.
        get_experiment_parameters (callable): Function which takes in the seed and experiment
            directory then produces all the parameters which get unpacked into the function
            responsible for running the experiment.
        log_level (int): Log level to use.
    Returns:
        (None): Saves all relevant files as it performs training.
    """
    
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    os.environ["TF_CPP_VMODULE"] = "gpu_process_state=10,gpu_cudamallocasync_allocator=10"

    paths.create_path(experiment_directory)

    # Prepare arguments for multiprocessing
    random_seeds = list(range(starting_seed, starting_seed + num_experiments))
    partial_get_experiment_parameters = functools.partial(
        get_experiment_parameters, directory=experiment_directory)
    experiment_args = [partial_get_experiment_parameters(
        seed) for seed in random_seeds]

    logging.basicConfig()
    logging.getLogger().setLevel(log_level)
    for args in experiment_args:
        experiment(**args)
        # Prevent GPU memory from fragmenting
        tf.keras.backend.clear_session()
        gc.collect()

def run_iterative_pruning_experiment(
    hyperparameters: Hyperparameters,
    random_seed: int,
    create_model: Callable[[], keras.models.Model],
    dataset: ds.Dataset,
    target_sparsity: float,
    sparsity_strategy: pruning.SparsityStrategy,
    pruning_rule: Callable,
    rewind_rule: Callable,
    global_pruning: bool = False,
    experiment_directory: str = './',
):
    """
    Function used to run the pruning experiements for a given random seed.
    Will perform iterative pruning given a specified pruning technique
    (e.g. Magnitude-based pruning) and a given rewind technique
    (e.g. Rewind to initial weights).

    Args:
        random_seed (int): Random seed for reproducability.
        create_model (callable): Function which produces the model.
        dataset (enum): Enum for the dataset being used.
        target_sparsity (float): Desired level of sparsity.
        sparsity_strategy (SparsityStrategy): Function which maps layer names
            to the appropriate level to sparsify them by.
        pruning_rule (callable, optional): Function used to prune model.
        rewind_rule (callable, optional): Function used for rewinding model weights.
        global_pruning (bool, optional): Boolean flag for whether pruning is done globally
            or layerwise. Defaults to False.
        experiment_directory (str): Path to place all experimental data.
    """
    # Set seed for reproducability
    utils.set_seed(random_seed)

    # Make models and save them
    model = create_model()
    mask_model = mod.create_masked_nn(create_model)

    mod.save_model(model, random_seed, 0, initial=True,
                   directory=experiment_directory)
    mod.save_model(mask_model, random_seed, 0, masks=True,
                   initial=True, directory=experiment_directory)

    pruning_step = 0
    make_sparsities = functools.partial(
        pruning.calculate_sparsity, sparsity_strategy=sparsity_strategy)
    while True:
        trial_data = train.train(
            random_seed=random_seed,
            pruning_step=pruning_step,
            model=model,
            mask_model=mask_model,
            dataset=dataset,
            hp=hyperparameters,
            output_directory=experiment_directory,
        )

        X_train, _, _, _ = dataset.load()
        iteration_count = np.sum(trial_data.train_accuracies != 0)

        logging.info(
            f'Took {iteration_count} iterations / {len(trial_data.train_accuracies)}')
        logging.info(
            f'Ended on epoch {np.ceil(iteration_count * hyperparameters.batch_size / X_train.shape[0])} out of {hyperparameters.epochs}')
        logging.info(
            'Best accuracies:\n'
            + f'Training: {np.max(trial_data.train_accuracies) * 100:.2f}%\n'
            + f'Validation: {np.max(trial_data.validation_accuracies) * 100:.2f}%\n'
        )

        # TODO: Have this differentiate based on outer layer name and kernel/bias
        layer_strings = [
            f'Layer {index}, '
            + f'Total Params: {total}, '
            + f'Nonzero Params: {nonzero}, '
            + f'Sparsity: {nonzero / total:%}'
            for index, (total, nonzero)
            in enumerate(utils.count_total_and_nonzero_params_per_layer(model))
        ]
        logging.info(
            'Layer sparsities:\n\t'
            + '\n\t'.join(layer_strings)
        )

        # Exit condition here makes sure we only break after having trained
        # with the final iteration sparser than the target
        if utils.model_sparsity(model) <= target_sparsity:
            break

        # Prune the model to the new sparsity and update the mask model
        pruning.prune(model, mask_model, pruning_rule,
                      make_sparsities, global_pruning=global_pruning)

        # Reset unpruned weights to original values.
        rewind.rewind_model_weights(model, mask_model, rewind_rule)

        # Testing conditions to verify correctness
        # initial = mod.load_model(random_seed, pruning_step=0, initial=True, directory=experiment_directory) 
        # rewound_weights = [w * mask for w, mask in zip(model.get_weights(), mask_model.get_weights())]
        # initial_weights = [w * mask for w, mask in zip(initial.get_weights(), mask_model.get_weights())]
        # assert all([(rewound == initial).all() for rewound, initial in zip(rewound_weights, initial_weights)])
        
        pruning_step += 1
