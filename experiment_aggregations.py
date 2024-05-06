"""
experiment_aggregations.py

Module containing functions for aggregating across experiment data. Basically just 
named functions applying some kind of Numpy aggregation within the context of values 
mapped from an ExperimentSummary object's experiments into a 2D array with the
dimension: `# Experiments, # Trials/Pruning Rounds`.

Author: Jordan Bourdeau
Date Created: 5/1/24
"""

import numpy as np

from src.harness import history
from src.metrics import trial_aggregations as t_agg

# ------------------------- Mean -------------------------

def mean_over_experiments(array: np.ndarray) -> float:
    return np.mean(array, axis=0)

def mean_over_trials(array: np.ndarray) -> float:
    return np.mean(array, axis=1)

def mean_overall(array: np.ndarray) -> float:
    return np.mean(array)

# ------------------------- Standard Deviation -------------------------

def std_over_experiments(array: np.ndarray) -> float:
    return np.std(array, axis=0)

def std_over_trials(array: np.ndarray) -> float:
    return np.std(array, axis=1)

def std_overall(array: np.ndarray) -> float:
    return np.std(array)

# ------------------------- Aggregated Over Experiments -------------------------

def aggregate_across_experiments(
    summary: history.ExperimentSummary, 
    trial_aggregation: callable, 
    experiment_aggregation: callable = mean_over_experiments,
    ) -> any:
    """
    Method used to aggregate over all the experiments within a summary
    using user-defined functions to aggregate trial and experiment data.
    
    Parameters:
        summary (callable): `ExperimentSummary` object being aggregated over.
        trial_aggregation (callable): Function which returns a single value when
            called on a `TrialData` object.
        experiment_aggregation (callable): Function which aggregates across all
            the data produced by aggregating over all the trial data.  

    Returns:
        any: Can return any type depending on how the aggregation is performed but
            will likely be a 1D array aggregating over all the trials from the 
            same pruning step.
    """
    trials_aggregated = t_agg.aggregate_across_trials(summary, trial_aggregation)
    experiment_aggregated = [experiment_aggregation(trials) for trials in trials_aggregated]
        
    return experiment_aggregated
    
def get_sparsities(summary: history.ExperimentSummary) -> list[float]:
    """
    Function used to retrieve the sparsities from an experiment summary.

    Args:
        summary (history.ExperimentSummary): `ExperimentSummary` object.

    Returns:
        list[float]: List of sparsities as percentages corresponding to each
            trial.
    """
    return aggregate_across_experiments(
        summary=summary,
        trial_aggregation=t_agg.get_sparsity_percentage,
        experiment_aggregation=mean_over_experiments,
    )