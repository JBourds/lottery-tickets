"""
plotting/seeding.py

Module containing plotting functions for tracking 
seeded weight initializations over training.

Author: Jordan Bourdeau
"""

from src.harness import architecture as arch
from src.harness import history
import src.harness.seeding as seed
from src.harness import utils
from src.metrics import trial_aggregations as t_agg
from src.metrics import experiment_aggregations as e_agg
from src.plotting import base_plots as bp

import copy
import functools
from importlib import reload
import itertools
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os
import tensorflow as tf
from typing import Any, Dict, Callable, Generator, Iterable, List, Tuple

WeightTrackingCallback = Tuple[
    str,
    Callable[
        [
            List[np.ndarray[bool]],
            history.TrialData,
        ],
        Any,
    ],
]

def prop_weights_in_mask(
    tracking_masks: List[np.ndarray], 
    trial: history.TrialData,
) -> List[float]:
    return [
        np.sum(mask[target]) / np.sum(target) 
        for target, mask in zip(tracking_masks, trial.masks)
    ]

def prop_positive_in_mask(
    tracking_masks: List[np.ndarray], 
    trial: history.TrialData,
) -> List[float]:
    return [
        np.sum(weights[target] >= 0) / np.sum(target) 
        for target, weights in zip(tracking_masks, trial.final_weights)
    ]

def layerwise_sparsity(
    masks: List[np.ndarray[bool]], 
    trial: history.TrialData
) -> List[float]:
    return [np.sum(m) / m.size for m in trial.masks]

def layerwise_positive(
    masks: List[np.ndarray[bool]], 
    trial: history.TrialData
) -> List[float]:
    return [np.sum(w >= 0) / w.size for w in trial.final_weights]

def trace_weights_over_time(
    tracking_masks: List[np.ndarray], 
    trials: Iterable[history.TrialData],
    callbacks: List[WeightTrackingCallback],
) -> Dict[str, np.ndarray]:
    data = {}
        
    for index, trial in enumerate(trials):
        for name, callback in callbacks:
            if data.get(name) is None:
                data[name] = []
            data[name].append(callback(tracking_masks, trial))
    return data

def compile_traced_weights(
    select_weights: seed.WeightsTarget,
    experiments: List[Generator[history.TrialData, None, None]],
    callbacks: List[WeightTrackingCallback],
) -> Dict:
    trial_values = []
    for experiment in experiments:
        trial = next(experiment)
        # Temp: Remove with new trials
        a = arch.Architecture(trial.architecture, trial.dataset)
        utils.set_seed(trial.random_seed)
        model = a.get_model_constructor()()
        initial_weights = model.get_weights()
        masks = [select_weights(w) for w in initial_weights]
        trials = itertools.chain([trial], experiment)
        results = trace_weights_over_time(masks, trials, callbacks)
        trial_values.append(results)
    return trial_values

def plot_seeded_vs_overall_positive(
    targets_2d: np.ndarray[np.ndarray[np.ndarray]],
    nontargets_2d: np.ndarray[np.ndarray[np.ndarray]],
    model_name: str,
    save_location: str = None,
):
    null_cols = [i for i, is_null in enumerate(np.isnan(targets_2d[0, 0])) if is_null]
    nontargets_2d = np.delete(nontargets_2d, null_cols, axis=2)
    targets_2d = np.delete(targets_2d, null_cols, axis=2)

    agg_mean_targets = np.mean(targets_2d, axis=0)
    agg_mean_actual = np.mean(nontargets_2d, axis=0)
    agg_std_targets = np.std(targets_2d, axis=0)
    agg_std_actual = np.std(nontargets_2d, axis=0)
    num_samples = targets_2d.shape[0]

    all_layers = [
        name for index, name 
        in enumerate(arch.Architecture.get_model_layers(model_name)) 
        if index not in null_cols
    ]

    plt.figure()
    plt.title(f"Overall Positive Proportion vs. Seeded Positive Proportion in {model_name}")
    plt.xlabel("Overall Layer Positive Proportion (%)")
    plt.ylabel("Seeded Layer Positive Proportion (%)")

    for index, label in enumerate(all_layers):
        bp.plot_aggregated_summary_ci(
            agg_mean_actual[:, index], 
            agg_mean_targets[:, index], 
            agg_std_targets[:, index], 
            num_samples,
            legend=label,
            show_ci_legend=False,
        )

    plt.legend()
    if save_location is not None:
        plt.savefig(save_location)
    plt.gca().invert_xaxis()
    plt.show()

def plot_seeded_vs_overall_sparsity(
    targets_2d: np.ndarray[np.ndarray[np.ndarray]],
    sparsity_2d: np.ndarray[np.ndarray[np.ndarray]],
    model_name: str,
    save_location: str = None,
):
    null_cols = [i for i, is_null in enumerate(np.isnan(targets_2d[0, 0])) if is_null]
    sparsity_2d = np.delete(sparsity_2d, null_cols, axis=2)
    targets_2d = np.delete(targets_2d, null_cols, axis=2)

    agg_mean_targets = np.mean(targets_2d, axis=0)
    agg_mean_actual = np.mean(sparsity_2d, axis=0)
    agg_std_targets = np.std(targets_2d, axis=0)
    agg_std_actual = np.std(sparsity_2d, axis=0)
    num_samples = targets_2d.shape[0]

    all_layers = [
        name for index, name 
        in enumerate(arch.Architecture.get_model_layers(model_name)) 
        if index not in null_cols
    ]

    plt.figure()
    plt.title(f"Overall Sparsity vs. Seeded Sparsity in {model_name}")
    plt.xlabel("Overall Layer Sparsity (%)")
    plt.ylabel("Seeded Layer Sparsity (%)")

    for index, label in enumerate(all_layers):
        bp.plot_aggregated_summary_ci(
            agg_mean_actual[:, index], 
            agg_mean_targets[:, index], 
            agg_std_targets[:, index], 
            num_samples,
            legend=label,
            show_ci_legend=False,
        )

    plt.legend()
    if save_location is not None:
        plt.savefig(save_location)
    plt.show()
