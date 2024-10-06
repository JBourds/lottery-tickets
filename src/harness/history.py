"""
history.py

Module containing class definitions for classes used to hold data throughout training.
Classes also provide interface to easily access calculations from aspects of training

Author: Jordan Bourdeay
Date Created: 4/28/24
"""

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Generator, List

import numpy as np

from src.harness import constants as C
from src.harness import mixins


@dataclass
class TrialData(mixins.PickleMixin, mixins.TimerMixin):
    """
    Class containing data from a single round of training.
    Includes a few metrics from training, the before/after
    weights, and the masks which were applied during training.
    Also includes the pruning step this was from.
    """
    pruning_step: int
    architecture: str
    dataset: str

    # Model parameters
    initial_weights: list[np.ndarray]
    final_weights: list[np.ndarray]
    masks: list[np.ndarray]

    # Metrics
    loss_before_training: float
    accuracy_before_training: float
    train_losses: np.array
    train_accuracies: np.array
    validation_losses: np.array
    validation_accuracies: np.array


def get_experiments(
    root: str,
    models_dir: str = C.MODELS_DIRECTORY, 
    eprefix: str = C.EXPERIMENT_PREFIX, 
    tprefix: str = C.TRIAL_PREFIX, 
    tdatafile: str = C.TRIAL_DATAFILE,
) -> List[Generator[TrialData, None, None]]: 
    def get_trials(epath: str) -> Generator[TrialData, None, None]:
        """
        Function which returns the loaded pieces of trial data in order from an
        experiment directory as a generator function.
        
        Assumes everything after the trial prefix is a number specifying the order.
        """
        trial_paths = [
            os.path.join(epath, path, tdatafile) for path in os.listdir(epath)
            if path.startswith(tprefix)
            and tdatafile in os.listdir(os.path.join(epath, path))
        ]
        
        for tpath in sorted(trial_paths, 
            key=lambda path: int(os.path.dirname(os.path.normpath(path)).split(tprefix)[-1])):
            yield TrialData.load_from(tpath)
        
    models_directory = os.path.join(root, models_dir)
    experiment_paths = [
        os.path.join(models_directory, path) for path in os.listdir(models_directory)
        if path.startswith(eprefix)
    ]
    return [get_trials(epath) for epath in experiment_paths]

