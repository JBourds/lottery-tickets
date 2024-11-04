"""
scripts/__init__.py

Module file containing utility functions for running scripts.

Author: Jordan Bourdeau
"""

import argparse
from functools import partial
import logging
import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from tensorflow import keras

from src.harness.architecture import Architecture, Hyperparameters
from src.harness import constants as C
from src.harness import dataset as ds
from src.harness import experiment
from src.harness import model as mod
from src.harness import pruning, rewind
from src.harness import seeding

def get_experiment_parameter_constructor(
    model: str,
    hyperparameters: Hyperparameters | None,
    dataset: str,
    rewind_rule: str,
    pruning_rule: str,
    seeding_rule: str | None,
    target_sparsity: float,
    sparsity_strategy: str,
    global_pruning: bool = False,
) -> Callable[[int, str], Dict[str, Any]]:
    """
    Generic function which takes experiment parameters and returns a function
    which acts as a constructor for the dictionary containing experimental
    parameters.
    """

    def inner_function(seed: int, directory: str) -> dict:
        """
        Inner function which inherits all the context it was created in but
        gets called with a unique seed and directory each time.
        """
        architecture = Architecture(model, dataset)
        make_model = architecture.get_model_constructor()

        return {
            'random_seed': seed,
            'create_model': make_model,
            'dataset': architecture.dataset,
            'target_sparsity': target_sparsity,
            'sparsity_strategy': get_sparsity_strategy(sparsity_strategy),
            'rewind_rule': get_rewind_rule(rewind_rule, seed=seed, directory=directory),
            'pruning_rule': get_pruning_rule(pruning_rule),
            'seeding_rule': get_seeding_rule(seeding_rule),
            'hyperparameters': hyperparameters,
            'global_pruning': global_pruning,
            'experiment_directory': directory,
        }

    return inner_function

# Functions which map command line arguments to internal representations


def get_log_level(log_level: int) -> int:
    match log_level:
        case 0:
            return logging.NOTSET
        case 1:
            return logging.DEBUG
        case 2:
            return logging.INFO
        case 3:
            return logging.WARNING
        case 4:
            return logging.ERROR
        case 5:
            return logging.CRITICAL
        case _:
            raise ValueError("Unknown log level '{log_level}'.")


def get_rewind_rule(rewind_rule: str, *args, **kwargs) -> Callable:
    match rewind_rule:
        case 'oi':
            return rewind.get_rewind_to_original_init_for(*args, **kwargs)
        case _:
            raise ValueError(
                f"'{rewind_rule}' is not a valid rewind rule option.")


def get_sparsity_strategy(sparsity_strategy: str) -> Callable[[str], float]:
    match sparsity_strategy.lower():
        case 'default':
            return pruning.default_sparsity_strategy
        case _:
            raise ValueError(
                f"'{sparsity_strategy}' is not a valid sparsity strategy option.")


def get_pruning_rule(pruning_rule: str) -> Callable:
    match pruning_rule:
        case 'lm':
            return pruning.low_magnitude_pruning
        case 'hm':
            return pruning.high_magnitude_pruning
        case _:
            raise ValueError(
                f"'{pruning_rule}' is not a valid pruning rule option.")

# Usage: <lm/hm/rand>-<% weights to seed>-<scale/set>-<value>
def get_seeding_rule(seeding_rule: str | None) -> Callable[[List[np.ndarray[float]]], None] | None:
    if seeding_rule is None:
        return None
    match = re.match('([a-zA-Z]+)-(\d{1,3})-([a-zA-z]+)-(\d+\.*\d*)$', seeding_rule)
    if match is None:
        raise ValueError('Invalid seeding rule string. Check usage.')
    target, proportion, transform, val = match.groups()
    proportion /= 100
    match target.lower():
        case 'hm':
            target = seeding.Target.HIGH
        case 'lm':
            target = seeding.Target.LOW
        case 'rand':
            target = seeding.Target.RANDOM
        case _:
            raise ValueError(f'Unsuported target: {target}')
    match transform.lower():
        case 'scale':
            transform = partial(seeding.scale_magnitude, factor=val)
        case 'set':
            transform = partial(seeding.set_to_constant, constant=val)
        case _:
            raise ValueError(f'Unsuported transform: {transform}')
    
    # This currently sets the same param for every layer
    return partial(seeding.seed_magnitude, targets=target, proportions=proportion, transforms=transform)
