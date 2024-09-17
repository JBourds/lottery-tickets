"""
experiment_aggregations.py

Module containing functions for aggregating across experiment data. 

Author: Jordan Bourdeau
"""

from typing import Any, Callable, Generator, List

import numpy as np

from src.harness import history
from src.metrics import trial_aggregations as t_agg

AggregatedTrials = List[List[Any]]
AggregatedExperiments = List[Any]

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
    summaries: List[Generator[history.TrialData, None, None]], 
    trial_aggregations: List[Callable[[history.TrialData], Any]], 
    experiment_aggregations: List[Callable[[AggregatedTrials], AggregatedExperiments]],
    ) -> List[AggregatedExperiments]:
    """
    Function which first aggregates over a 2D array of dimensions `N` x `M` where `N` is the
    number of experiments and `M` are the number of trials (rounds of iterative pruning)
    and applies each aggregating function specified in the list of trial aggregations. This
    step will return an `A` x `N` x `M` array where `A` is equal to the number of trial
    aggregations performed, and the type of each list is whatever the type output by the trial
    aggregation. Then, an experiment aggregation is applied over corresponding values from
    each trial across experiments, effectively collapsing the second dimension and outputting
    a `B` x `A` x `M` dimension array, where `B` is the number of experiment aggregations.
    """
    trials_aggregated = t_agg.aggregate_across_trials(summaries, trial_aggregations)
    aggregated_data = []
    for e_agg in experiment_aggregations:
        aggregated_data.append([e_agg(trial_agg) for trial_agg in trials_aggregated])
    return aggregated_data
