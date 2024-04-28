"""
experiment.py

Module containing code for actually running the lottery ticket hypothesis experiemnts.

Modified By: Jordan Bourdeau
Date: 3/17/24
"""

import functools
import numpy as np

import src.harness.constants as C
from src.harness.training import TrainingRound

class ExperimentData:
  def __init__(self):
    """
    Class which stores the data from an experiment 
    (list of `TrainingRound` objects which is of length N, where N is the # of pruning steps).
    """
    self.pruning_rounds: list[TrainingRound] = []

  def add_pruning_round(self, round: TrainingRound):
    """
    Method used to add a `TrainingRound` object to the internal representation.

    :param round: `TrainingRound` object being added.
    """
    self.pruning_rounds.append(round)


class ExperimentSummary:
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

  def __str__(self) -> str:
    """
    String representation to create a summary of the experiment.

    :returns: String representation.
    """
    for seed, experiment in self.experiments.items():
      print(f'\nSeed {seed}')
      for idx, round in enumerate(experiment.pruning_rounds):
        print(f'Pruning Step {idx}:')
        print(round)

