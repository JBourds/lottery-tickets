"""
history.py

Module containing class definitions for classes used to hold data throughout training.
Classes also provide interface to easily access calculations from aspects of training

Author: Jordan Bourdeay
Date Created: 4/28/24
"""

from dataclasses import dataclass
import numpy as np
import os
import sys

from src.harness import mixins
from src.metrics.experiment_aggregations import mean_over_experiments

@dataclass
class TrialData(mixins.PickleMixin):
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
    
    def get_loss_before_training(self) -> float:
        """
        Returns:
            float: Model loss on the masked initial weights.
        """
        return self.loss_before_training
    
    def get_accuracy_before_training(self) -> float:
        """
        Returns:
            float: Model accuracy on the masked initial weights.
        """
        return self.accuracy_before_training
    
    def get_sparsity(self) -> float:
        """
        Calculate the sparsity of the model at this particular training round.

        Returns:
            float: The sparsity ratio of enabled parameters to total parameters.
        """
        enabled_parameter_count: int = np.sum([np.sum(mask) for mask in self.masks])
        total_parameter_count: int = np.sum([np.size(mask) for mask in self.masks])
        return enabled_parameter_count / total_parameter_count
    
    def get_best_accuracy(self, use_test: bool = True) -> float:
        """
        Get the best accuracy achieved during training or testing.

        Args:
            use_test (bool, optional): Whether to use test accuracies. Defaults to True.

        Returns:
            float: The highest accuracy achieved.
        """
        return np.max(self.test_accuracies if use_test else self.train_accuracies)
        
    def get_best_loss(self, use_test: bool = True) -> float:
        """
        Get the best loss achieved during training or testing.

        Args:
            use_test (bool, optional): Whether to use test losses. Defaults to True.

        Returns:
            float: The lowest loss achieved.
        """
        return np.max(self.test_losses if use_test else self.train_losses)
        
    def get_early_stopping_iteration(self) -> int:
        """
        Get the step at which early stopping occurred during training.

        Returns:
            int: The step at which training was stopped early.
        """
        performance_evaluation_frequency: int = self.train_accuracies.shape[0] // self.test_accuracies.shape[0]
        nonzero_indices = np.nonzero(self.train_accuracies == 0)[0]
        stop_index: int = len(self.train_accuracies) if len(nonzero_indices) == 0 else nonzero_indices[0]
        return stop_index * performance_evaluation_frequency
    
    def get_pruning_step(self)-> int:
        """
        Get the pruning step of the TrialData.

        returns 
            int: The trials respective pruning step.
        """
        return self.pruning_step
    
    def __str__(self):
        """
        Returns:
            str: String representation of a training round.
        """
        representation: str = f"""Pruning Step {self.pruning_step}
        Sparsity: {self.get_sparsity() * 100:.3f}%
        Best Training Accuracy: {self.get_best_accuracy(use_test=False) * 100:.3f}%
        Best Test Accuracy: {self.get_best_accuracy() * 100:.3f}%
        Best Training Loss: {self.get_best_loss(use_test=False):.3f}
        Best Test Loss: {self.get_best_loss():.3f}
        Early Stopping Iteration: {self.get_early_stopping_iteration()}
        """
        return representation


class ExperimentData(mixins.PickleMixin):
    
    def __init__(self):
        """
        Class which stores the data from an experiment 
        (list of `TrialData` objects which is of length N, where N is the # of pruning steps).
        """
        self.pruning_rounds: list[TrialData] = []

    def add_pruning_round(self, round: TrialData):
        """
        Method used to add a `TrialData` object to the internal representation.

        :param round: `TrialData` object being added.
        """
        self.pruning_rounds.append(round)

    def __str__(self) -> str:
      """
      String representation to create a summary of an experiment.

      :returns: String representation.
      """
      return '\n'.join([str(round) for round in self.pruning_rounds])


class ExperimentSummary(mixins.PickleMixin):
    
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
    
    def aggregate_across_experiments(
        self, 
        trial_aggregation: callable, 
        experiment_aggregation: callable = mean_over_experiments,
        ) -> any:
        """
        Method used to aggregate over all the experiments within a summary
        using user-defined functions to aggregate trial and experiment data.
        
        Parameters:
            trial_aggregation (callable): Function which returns a single value when
                called on a `TrialData` object.
            experiment_aggregation (callable): Function which aggregates across all
                the data produced by aggregating over all the trial data.  

        Returns:
            any: Can return any type depending on how the aggregation is performed but
                will likely be a 1D array aggregating over all the trials from the 
                same pruning step.
        """
        trials_aggregated: dict = {}
        experiment_aggregated: list[np.array] = []
        # Iterate across experiments
        for experiment in self.experiments.values():
            # Iterate over trials within an experiment
            for trial in experiment.pruning_rounds:
                # Check if the 
                if trial.get_pruning_step() in trials_aggregated.keys():
                    trials_aggregated.get(trial.get_pruning_step()).append((trial_aggregation(trial)))
                else:
                    trials_aggregated[trial.get_pruning_step()] = [trial_aggregation(trial)]
                    
        for trial in trials_aggregated.keys():
            experiment_aggregated.append(experiment_aggregation((trials_aggregated[trial])))
            
        return experiment_aggregated
            
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


