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
from typing import Union

from src.harness import mixins
from src.metrics.experiment_aggregations import mean_over_experiments

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

        Returns: 
            int: The trials respective pruning step.
        """
        return self.pruning_step
    
    def get_positive_weight_ratio(
        self, 
        layerwise: bool = False, 
        use_initial_weights: bool = False
        ) -> Union[float, list[float]]:
        """
        Function used to get the positive weight ratio, either globally across all layers or
        layerwise.

        Returns:
            Union[float, list[float]]: Single float or list of floats depending on whether it performs
                calculations globally or by layer.
        """
        return self._perform_operation_globally_or_layerwise(
            operation=self._get_ratio_of_unmasked_positive_weights,
            layerwise=layerwise,
            use_initial_weights=use_initial_weights
        )
    
    def get_average_magnitude(
        self, 
        layerwise: bool = False, 
        use_initial_weights: bool = False,
        ) -> Union[float, list[float]]:
        """
        Function used to get the average magnitude of parameters, either 
        globally across all layers or layerwise.

        Returns:
            Union[float, list[float]]: Single float or list of floats depending on whether it performs
                calculations globally or by layer.
        """
        return self._perform_operation_globally_or_layerwise(
            operation=self._get_average_parameter_magnitude,
            layerwise=layerwise,
            use_initial_weights=use_initial_weights
        )
    
    # ------------------------- Private Helper Methods -------------------------
    
    def _perform_operation_globally_or_layerwise(
        self, 
        operation: callable,
        layerwise: bool = False, 
        use_initial_weights: bool = False,
        ) -> any:
        """
        Function used to get the average magnitude of parameters, either 
        globally across all layers or layerwise.

        Args:
            operation (callable): Function which is being performed layerwise of globally.
            layerwise (bool, optional): Get proportion for each layer. Defaults to False.
            use_initial_weights (bool, optional): Use initial, masked weights or final trained weights. 
                Defaults to False.

        Returns:
            any: Returns type depends on the operation.
        """
        # Convert masks into boolean Numpy array for indexing, and convert weights into Numpy array as well
        masks: list[np.ndarray] = [mask.astype('bool') for mask in self.masks]
        weights: list[np.ndarray] = self.initial_weights if use_initial_weights else self.final_weights
        weights = [w for w in weights]
        
        if layerwise:
            # Since the private method expects a list, just make each entry into a list of 1 element
            masks = [[m] for m in masks]
            weights = [[w] for w in weights]
            return list(map(operation, zip(weights, masks)))
        
        return operation(weights, masks)

    def _get_ratio_of_unmasked_positive_weights(self, weights: list[np.ndarray], masks: list[np.ndarray]) -> float:
        """
        Private function which computers the proportion of unmasked positive weights.
        
        Args:   
            weights: (list[np.ndarray]): List of Numpy arrays for the weights in each layer being included.
            masks: (list[np.ndarray]): List of Numpy arrays for the masks in each layer being included.
            
        Returns:
            float: Proportion of postive parameters in the unmasked weights.
        """
        assert len(weights) == len(masks), 'Weight and mask arrays must be the same length.'
        assert np.all([w.shape == m.shape for w, m in zip(weights, masks)]), 'Weights and masks must all be the same shape'
        
        # There is a chance weights could be set to 0 but not masked (e.g. bias terms)
        # Because of this, we use the mask as indices and only consider the unmasked portion 
        total_positive: float = np.sum([np.sum(w[mask] >= 0) for w, mask in zip(weights, masks)])
        total_nonzero: float = np.sum([np.size(w[mask]) for w, mask in zip(weights, masks)])
        
        return total_positive / total_nonzero
    
    def _get_average_parameter_magnitude(self, weights: list[np.ndarray], masks: list[np.ndarray]) -> float:
        """
        Private function which computers the average magnitude in a list of unmasked weights.
        
        Args:   
            weights: (list[np.ndarray]): List of Numpy arrays for the weights in each layer being included.
            masks: (list[np.ndarray]): List of Numpy arrays for the masks in each layer being included.
                
        Returns:
            float: Average parameter magnitude of the unmasked weights.
        """
        assert len(weights) == len(masks), 'Weight and mask arrays must be the same length.'
        assert np.all([w.shape == m.shape for w, m in zip(weights, masks)]), 'Weights and masks must all be the same shape'
        
        unmasked_weights: list[np.ndarray] = [w[mask] for w, mask in zip(weights, masks)]
        unmasked_weight_sum_magnitude: float = np.sum([np.sum(np.abs(w)) for w in unmasked_weights])
        unmasked_weight_count: int = np.sum([np.size(w) for w in unmasked_weights])
        
        return unmasked_weight_sum_magnitude / unmasked_weight_count
    
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
        self.trials[trial.get_pruning_step()] = trial

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
            for trial in experiment.trials.values():
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
          for idx, round in enumerate(experiment.trials.values()):
              print(f'Pruning Step {idx}:')
              print(round)


