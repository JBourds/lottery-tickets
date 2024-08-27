"""
scripts/lenet_300_100.py

Base script which all additional training scripts are based on.

Author: Jordan Bourdeau
Date Created: 4/30/24
"""

import argparse
import functools
import logging
import os
import sys
from typing import List

import numpy as np

from scripts import get_experiment_parameter_constructor, get_log_level
from src.harness import constants as C
from src.harness import dataset as ds
from src.harness import experiment
from src.harness import model as mod
from src.harness import pruning, rewind
from src.harness.architecture import Hyperparameters


def run_parallel_experiments(
    experiment_directory: str,
    starting_seed: int = 0,
    num_experiments: int = 1,
    num_batches: int = 1,
    target_sparsity: float = 0.85,
    sparsity_strategy: str = 'default',
    model: str = 'lenet',
    hyperparameters: Hyperparameters | None = None,
    dataset: str = 'mnist',
    rewind_rule: str = 'oi',
    pruning_rule: str = 'lm',
    max_processes: int | None = None,
    log_level: int = 2,
    global_pruning: bool = False
) -> None:
    """
    Run parallelized experiments with specified configurations.

    Parameters
    ----------
    experiment_directory: str
        Directory to store all models and experiment summaries.
    starting_seed : int, optional
        The initial seed for random number generation. Defaults to 0.
    num_experiments : int, optional
        Number of experiments to run. Defaults to 1.
    num_batches : int, optional
        Number of batches to split the experiments into. Defaults to 1.
    target_sparsity : float
        Desired sparsity overall. 
    sparsity_strategy : SparsityStrategy
        Function which maps a layer's name to the amount to prune by.
    model : str, optional
        Model architecture to use. Defaults to 'lenet'.
    hyperparameters : Hyperparameters, optional
        Hyperparameters to use. Will default to defaults for provided model.
    dataset : str, optional
        Dataset to use for training. Defaults to 'mnist'.
    rewind_rule : str, optional
        Rule for rewinding weights. Defaults to 'oi' (original initialization).
    pruning_rule : str, optional
        Rule for pruning. Defaults to 'lm' (low magnitude pruning).
    max_processes : int, optional
        Maximum number of parallel processes. Defaults to the number of CPU cores.
    log_level : int, optional
        Logging level. Defaults to 2 (Info).
    global_pruning : bool, optional
        Whether to use global pruning (True) or layerwise pruning (False). Defaults to False.

    Notes
    -----
    This function sets up the necessary parameters and runs the experiments in parallel batches.
    Each batch runs a subset of the total experiments, with seeds adjusted for each batch.

    The experiments are saved in the specified output directory, which is generated based on the 
    model name if not provided.
    """
    if max_processes is None:
        max_processes = os.cpu_count()

    # Construct the full experiment directory path
    experiment_directory = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        C.EXPERIMENTS_DIRECTORY,
        experiment_directory if experiment_directory else model
    )

    get_experiment_parameters = get_experiment_parameter_constructor(
        model=model,
        hyperparameters=hyperparameters,
        dataset=dataset,
        rewind_rule=rewind_rule,
        pruning_rule=pruning_rule,
        target_sparsity=target_sparsity,
        sparsity_strategy=sparsity_strategy,
        global_pruning=global_pruning
    )

    # Perform parallelized training in evenly split batches
    num_experiments_in_batch = int(np.ceil(num_experiments / num_batches))
    for batch_idx in range(num_batches):
        batch_starting_seed = starting_seed + batch_idx * num_experiments_in_batch
        batch_directory = os.path.join(
            experiment_directory, f'batch_{batch_idx}')

        # Last batch could be smaller
        if batch_idx == num_batches - 1:
            num_experiments_in_batch = num_experiments - \
                batch_idx * num_experiments_in_batch

        experiment.run_experiments(
            starting_seed=batch_starting_seed,
            num_experiments=num_experiments_in_batch,
            experiment_directory=batch_directory,
            experiment=experiment.run_iterative_pruning_experiment,
            get_experiment_parameters=get_experiment_parameters,
            max_processes=max_processes,
            log_level=get_log_level(log_level),
        )
