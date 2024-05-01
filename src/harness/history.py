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
from src.harness import mixins

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
        
    def get_pruning_rounds(self):
        return self.pruning_rounds

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
    
    def aggregate_across_experiments(self,agg_trial:callable, agg_exp:callable = np.mean) -> dict[int: list[float]]:
        """
        Method that reads in the data from each experiment and aggregates

        :param agg_trial: the method used to aggregate all the trial data   
        :param agg_exp:   the method used to aggregate the experiment data
        """
        trials_aggregated = {}
        experiment_aggregated = []
        for experiment in self.experiments.values():
            for trial in experiment.get_pruning_rounds():
                if trial.get_pruning_step() in trials_aggregated.keys():
                    trials_aggregated.get(trial.get_pruning_step()).append((agg_trial(trial)))
                else:
                    trials_aggregated[trial.get_pruning_step()] = [agg_trial(trial)]
        for trial in trials_aggregated.keys():
            experiment_aggregated.append(agg_exp((trials_aggregated[trial])))
            
        return experiment_aggregated

    def get_sparcity(self, trial: TrialData):
        return trial.get_sparsity() * 100

    def early_stop(self,trial: TrialData):
        return trial.get_early_stopping_iteration()
    
    def accuracy_at_stop(self,trial:TrialData):
        return np.max(trial.test_accuracies)
        
    def get_test_accuracy(self,trial:TrialData):
        return trial.test_accuracies[trial.test_accuracies != 0]
        
    def get_positive_ratio_total_final(self, trial:TrialData):
        total_positive = np.sum([np.sum(weights[weights != 0] > 0) for weights in trial.final_weights])
        total_nonzero = np.sum([np.sum(weights != 0) for weights in trial.final_weights])
        return total_positive / total_nonzero
        
    def print_acc(self,trial):
        print(trial.test_accuracies)
    
    def get_negative_ratio_total_final(self,trial:TrialData):
        return 1 - self.get_positive_ratio_total_final(trial)
        
    def get_positive_ratio_total_init(self, trial:TrialData):
        total_positive = np.sum([np.sum(weights[weights != 0] > 0) for weights in trial.initial_weights])
        total_nonzero = np.sum([np.sum(weights != 0) for weights in trial.initial_weights])
        return total_positive / total_nonzero
    
    def get_negative_ratio_total_init(self, trial:TrialData):
        return 1 - get_positive_ratio_total_init(trial)
    
    def mean_list(self, iter):
        return np.mean(iter,axis = 0)
            
            
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


