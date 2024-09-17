"""
scripts/python/base.py

Base functionality used to generate plots
from external script.

Author: Jordan Bourdeau
"""

import functools
from importlib import reload
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os
from typing import Generator

from src.harness import constants as C
from src.harness import history
from src.harness import paths
from src.metrics import experiment_aggregations as e_agg
from src.metrics import trial_aggregations as t_agg

from src.plotting import base_plots as bp
from src.plotting import global_plots as gp

    
def make_plots(
    root: str,
    models_dir: str = C.MODELS_DIRECTORY,
    plots_dir: str = C.PLOTS_DIRECTORY,
    eprefix: str = C.EXPERIMENT_PREFIX,
    tprefix: str = C.TRIAL_PREFIX,
    tdata: str = C.TRIAL_DATAFILE,
):
    """
    Function which creates all the plots for an experiment given information about
    its root directory and naming scheme.

    @param root (str): Directory where all the individual experiments were run.
    @param models_directory (str): Directory where models are put in relative to the root.
    @param plots_dir (str): Directory to put the plots in relative to the root.
    @param eprefix (str): String prefix for individual experiments (random seeds).
    @param tprefix (str): String prefix for individual trials (rounds of IMP).
    @param tdata (str): Name of the pickled trial data file within the directory.

    @returns (None): Saves plots to specified directory.
    """
    experiments = history.get_experiments(root, models_dir, eprefix, tprefix, tdata)
    trial_aggregations = [
        ('pruning_step', t_agg.get_pruning_step),
        ('loss_before_training', t_agg.get_loss_before_training),
        ('acc_before_training', t_agg.get_accuracy_before_training),
        ('global_pos_percent', t_agg.get_global_percent_positive_weights),
        ('layer_pos_percent', t_agg.get_layerwise_percent_positive_weights),
        ('global_avg_mag', t_agg.get_global_average_magnitude),
        ('layer_avg_mag', t_agg.get_layerwise_average_magnitude),
        ('best_val_acc', t_agg.get_best_accuracy_percent),
        ('best_val_loss', t_agg.get_best_loss),
        ('sparsity', t_agg.get_sparsity_percentage),
        ('stop_iter', t_agg.get_early_stopping_iteration),
    ]
    experiment_aggregations = [
        ('mean', e_agg.mean_over_experiments),
        ('std', e_agg.std_over_experiments), 
        ('0th', e_agg.nth_experiment),
        ('num_samples', e_agg.num_samples),
    ]
    results = {}
    t_functions = [f for _, f in trial_aggregations]
    e_functions = [f for _, f in experiment_aggregations]
    data = e_agg.aggregate_across_experiments(experiments, t_functions, e_functions)
    for ((e_name, _), e_data) in zip(experiment_aggregations, data):
        results[e_name] = {}
        for ((t_name, _), t_data) in zip(trial_aggregations, e_data):
            results[e_name][t_name] = t_data 
    
    plot_params = [
        {'name': 'stop_iter', 'x': ('0th', 'pruning_step'), 'func': gp.plot_early_stopping},
        {'name': 'global_pos_percent', 'x': ('0th', 'pruning_step'), 'func': gp.plot_sign_proportion},
        {'name': 'best_val_acc', 'x': ('0th', 'pruning_step'), 'func': gp.plot_best_accuracy_at_early_stopping},
        {'name': 'loss_before_training', 'x': ('0th', 'pruning_step'), 'func': gp.plot_loss_before_training},
        {'name': 'acc_before_training', 'x': ('0th', 'pruning_step'), 'func': gp.plot_accuracy_before_training},
    ] 

    for params in plot_params:
        save_location = os.path.join(root, plots_dir, params['name'])
        e_key, t_key = params['x']
        x = results[e_key][t_key]
        y_mean = results['mean'][params['name']]
        y_std = results['std'][params['name']]
        num_samples = results['num_samples'][params['name']]
        params['func'](x=x, y_mean=y_mean, y_std=y_std, num_samples=num_samples, save_location=save_location)
         
