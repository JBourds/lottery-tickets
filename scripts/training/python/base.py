"""
scripts/lenet_300_100.py

Base script which all additional training scripts are based on.

Author: Jordan Bourdeau
Date Created: 4/30/24
"""

import argparse
import datetime
import functools
import logging
import os
import sys
from typing import List

import numpy as np

from scripts.training.python import get_experiment_parameter_constructor, get_log_level
from src.harness import constants as C
from src.harness import dataset as ds
from src.harness import experiment
from src.harness import model as mod
from src.harness import pruning, rewind
from src.harness.architecture import Hyperparameters


def run_experiments(
    path: str,
    starting_seed: int = 0,
    num_experiments: int = 1,
    target_sparsity: float = 0.85,
    sparsity_strategy: str = 'default',
    model: str = 'lenet',
    hyperparameters: Hyperparameters | None = None,
    dataset: str = 'mnist',
    rewind_rule: str = 'oi',
    pruning_rule: str = 'lm',
    log_level: int = 2,
    global_pruning: bool = False
) -> None:
    """
    Run parallelized experiments with specified configurations.

    Parameters
    ----------
    path: str
        Directory to store all models and experiment summaries.
    starting_seed : int, optional
        The initial seed for random number generation. Defaults to 0.
    num_experiments : int, optional
        Number of experiments to run. Defaults to 1.
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
    log_level : int, optional
        Logging level. Defaults to 2 (Info).
    global_pruning : bool, optional
        Whether to use global pruning (True) or layerwise pruning (False). Defaults to False.
    """
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

    experiment.run_experiments(
        starting_seed=starting_seed,
        num_experiments=num_experiments,
        experiment_directory=path,
        experiment=experiment.run_iterative_pruning_experiment,
        get_experiment_parameters=get_experiment_parameters,
        log_level=get_log_level(log_level),
    )