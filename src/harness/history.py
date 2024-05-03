"""
history.py

Module containing class definitions for classes used to hold data throughout training.
Classes also provide interface to easily access calculations from aspects of training

Author: Jordan Bourdeay
Date Created: 4/28/24
"""

from dataclasses import dataclass
from datetime import datetime
import numpy as np
import os
import sys

from src.harness import mixins

@dataclass
class TrialData(mixins.PickleMixin, mixins.TimerMixin):
    """
    Class containing data from a single round of training.

    Parameters:
    :param pruning_step:    (int) Integer for the step in pruning. 
    :param initial_weights: (list[np.ndarray]) Initial weights of the model.
    :param final_weights:   (list[np.ndarray]) Final weights of the model.
    :param masks:           (list[np.ndarray]) List of mask model weights (binary mask).
    """
    pruning_step: int
    
    # Model parameters
    initial_weights: list[np.ndarray]
    final_weights: list[np.ndarray]
    masks: list[np.ndarray]
    
    # Metrics
    loss_before_training: float
    accuracy_before_training: float
    train_losses: np.array
    train_accuracies: np.array
    test_losses: np.array
    test_accuracies: np.array
    
    def __str__(self):
        """
        Returns:
            str: String representation of a training round.
        """
        representation: str = f'Trial from puning step {self.pruning_step}'
        return representation


class ExperimentData(mixins.PickleMixin, mixins.TimerMixin):
    
    def __init__(self):
        """
        Class which stores the data from an experiment 
        (list of `TrialData` objects which is of length N, where N is the # of pruning steps).
        """
        self.trials: dict[int: TrialData] = {}
        
    def get_trial_count(self) -> int:
        """
        Function to return the number of trials in an experiment.

        Returns:
            int: Number of trials.
        """
        return len(self.trials)
        
    def add_trials(self, trials: list[TrialData]):
        """
        Method used to add a list `TrialData` objects to the internal representation.

        :param trials: List of `TrialData` objects being added.
        """
        for trial in trials:
            self.add_trial(trial)

    def add_trial(self, trial: TrialData):
        """
        Method used to add a `TrialData` object to the internal representation.

        :param trial: `TrialData` object being added.
        """
        self.trials[trial.pruning_step] = trial

    def __str__(self) -> str:
      """
      String representation to create a summary of an experiment.

      :returns: String representation.
      """
      return '\n'.join([str(round) for round in self.trials.values()])


class ExperimentSummary(mixins.PickleMixin, mixins.TimerMixin):
    
    def __init__(self):
        """
        Class which stores data from many experiments in a dictionary, where the key
        is the random seed used for the experiment and the value is an `ExperimentData` object.
        """
        self.experiments: dict[int: ExperimentData] = {}

    def add_experiment(self, seed: int, experiment: ExperimentData):
        """
        Method to add a new experiment to the internal dictionary.

        :param seed:        Integer for the random seed used in the experiment.
        :param experiment: `Experiment` object to store.
        """
        self.experiments[seed] = experiment
        
    def get_experiment_count(self) -> int:
        """
        Function to return the number of experiments in a summary.

        Returns:
            int: Number of experiments.
        """
        return len(self.experiments)
            
    def __str__(self) -> str:
      """
      String representation to create a summary of the experiment.

      :returns: String representation.
      """
      for seed, experiment in self.experiments.items():
          print(f'\nSeed {seed}')
          for idx, round in enumerate(experiment.trials.values()):
              print(f'Pruning Step {idx}:')
              print(round)


