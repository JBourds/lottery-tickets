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

# --------------------- Mean ---------------------

def mean_over_experiments(array: np.ndarray) -> float:
    return np.mean(array, axis=0)

def mean_over_trials(array: np.ndarray) -> float:
    return np.mean(array, axis=1)

def mean_overall(array: np.ndarray) -> float:
    return np.mean(array)

# --------------------- Standard Deviation ---------------------

def std_over_experiments(array: np.ndarray) -> float:
    return np.std(array, axis=0)

def std_over_trials(array: np.ndarray) -> float:
    return np.std(array, axis=1)

def std_overall(array: np.ndarray) -> float:
    return np.std(array)