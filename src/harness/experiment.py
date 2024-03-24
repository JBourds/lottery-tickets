# Copyright (C) 2018 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
experiment.py

Module containing code for actually running the lottery ticket hypothesis experiemnts.

Modified By: Jordan Bourdeau
Date: 3/17/24
"""

import functools
import numpy as npß

from src.harness.constants import Constants as C
from src.harness.dataset import load_and_process_mnist
from src.harness.model import LeNet300
from src.harness.training import train, TrainingRound
from src.harness.pruning import prune_by_percent

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


def run_experiments(num_models: int = C.NUM_MODELS, 
                    training_iterations: int = C.TRAINING_ITERATIONS, 
                    pruning_percents: dict[int: float] = C.PRUNING_PERCENTS,
                    pruning_steps: int = C.PRUNING_STEPS
                    ) -> ExperimentSummary:
    """
    Function used to run the full lottery ticket experiment.

    :param num_models:          Integer for the number of models to produce.
    :param training_iterations: Integer for the number of iterations to use for each round of training.
    :param pruning_percents:    Dictionary mapping pruning step index to % of masks to prune.
    :param pruning_steps:       Integer for the number of pruning steps to perform.

    :returns: Returns an `ExperimentSummary` object with data from every experiment and round of training.
    """

    assert num_models > 0 and isinstance(num_models, int), 'Must have integer number of models greater than 0.'
    assert training_iterations > 0 and isinstance(training_iterations, int), 'Must have integer number of training iterations greater than 0.'
    assert pruning_steps >= 0 and isinstance(pruning_steps, int), 'Must have integer number of pruning steps greater than or equal to 0.'

    make_dataset: callable = load_and_process_mnist
    train_model: callable = functools.partial(train, iterations=training_iterations)
    prune_masks: callable = functools.partial(prune_by_percent, pruning_percents)

    summary: ExperimentSummary = ExperimentSummary()

    for seed in range(num_models):
        make_model: callable = functools.partial(LeNet300, seed)
        data: ExperimentData = experiment(make_dataset, make_model, train_model, prune_masks, pruning_steps)
        summary.add_experiment(seed, data)
    
    return summary


def experiment(make_dataset: callable, 
               make_model: callable, 
               train_model: callable, 
               prune_masks: callable, 
               pruning_steps: int,
               presets: dict[str: np.array] = None) -> ExperimentData:
  """
  Run the lottery ticket experiment for the specified number of iterations.

  :param seed:          Random seed/unique identifer to use to identify the experiment.
  :param make_dataset:  A function that, when called with no arguments, will create the training and test sets.
  :param make_model:    A function that, when called with four arguments (input_tensor,
                        label_tensor, presets, masks), creates a model object that descends from
                        model_base. Presets and masks are optional.
  :param train_model:   A function that, when called with three arguments (pruning iteration number, 
                        tuple with dataset training/test sets, model), trains the model using the
                        dataset and returns the model's initial and final weights as dictionaries.
  :param prune_masks:   A function that, when called with two arguments (dictionary of
                        current masks, dictionary of final weights), returns a new dictionary of
                        masks that have been pruned. Each dictionary key is the name of a tensor
                        in the network; each value is a numpy array containing the values of the
                        tensor (1/0 values for mask, weights for the dictionary of final weights).
  :param pruning_steps: The number of pruning iterations to perform.
  :param presets:       (optional) The presets to use for the first iteration of training.
                        In the form of a dictionary where each key is the name of a tensor and
                        each value is a numpy array of the values to which that tensor should
                        be initialized.

  :returns: `TrainingSummary` object which contains information from all the rounds of training.
  """

  experiment: ExperimentData = ExperimentData()

  # A helper function that trains the network once according to the behavior
  # determined internally by the+ train_model function.
  def train_once(pruning_step: int, presets=None, masks=None):
    print(f'Pruning Step {pruning_step}')
    X_train, Y_train, _, _ = make_dataset()
    model: LeNet300 = make_model(X_train, Y_train, presets=presets, masks=masks)
    return train_model(make_dataset, model, pruning_step)

  # Run once normally.
  round: TrainingRound = train_once(0, presets=presets)
  experiment.add_pruning_round(round)
  initial_weights: dict[str: np.ndarray] = round.initial_weights

  # Create the initial masks with no weights pruned.
  masks = {}
  for k, v in initial_weights.items():
    masks[k] = np.ones(v.shape)

  # Begin the training loop.
  for pruning_step in range(1, pruning_steps + 1):
    # Prune the network.
    masks = prune_masks(masks, round.final_weights)
    # Train the network again.
    round = train_once(pruning_step, presets=initial_weights, masks=masks)
    experiment.add_pruning_round(round)

  return experiment
