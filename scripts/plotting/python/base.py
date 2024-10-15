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
from src.plotting import layerwise_plots as lp

    
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
    trial_aggregations = {
        'pruning_step': t_agg.get_pruning_step,
        'loss_before_training': t_agg.get_loss_before_training,
        'acc_before_training': t_agg.get_accuracy_before_training,
        'layer_names': t_agg.get_layer_names,
        'best_val_acc': t_agg.get_best_accuracy_percent,
        'best_val_loss': t_agg.get_best_loss,
        'sparsity': t_agg.get_sparsity_percentage,
        'early_stopping': t_agg.get_early_stopping_iteration,

        # Compare the masked/unmasked initial weights against each other to see if there is a trend from the initialization
        # in addition to what the values do once trained
        
        'final_positive_percent': t_agg.get_global_percent_positive_weights,
        'initial_positive_percent': functools.partial(t_agg.get_global_percent_positive_weights, use_initial_weights=True),
        'masked_initial_positive_percent': functools.partial(t_agg.get_global_percent_positive_weights, use_initial_weights=True, use_masked_weights=True),

        'final_avg_mag': t_agg.get_global_average_magnitude,
        'initial_avg_mag': functools.partial(t_agg.get_global_average_magnitude, use_initial_weights=True),
        'masked_initial_avg_mag': functools.partial(t_agg.get_global_average_magnitude, use_initial_weights=True, use_masked_weights=True),
        
        # Layerwise metrics
        'layer_pos_percent': t_agg.get_layerwise_percent_positive_weights,
        'layer_avg_mag': t_agg.get_layerwise_average_magnitude,
    }
    experiment_aggregations = {
        'mean': e_agg.mean_over_experiments,
        'std': e_agg.std_over_experiments, 
        '0th': e_agg.nth_experiment,
        'num_samples': e_agg.num_samples,
        'values': lambda x: x,
    }
    results = {}
    t_functions = [f for f in trial_aggregations.values()]
    e_functions = [f for f in experiment_aggregations.values()]
    data = e_agg.aggregate_across_experiments(experiments, t_functions, e_functions)
    for e_name, e_data in zip(experiment_aggregations.keys(), data):
        results[e_name] = {}
        for t_name, t_data in zip(trial_aggregations.keys(), e_data):
            results[e_name][t_name] = t_data 

    plot_params = [
        {'name': 'early_stopping', 'x': ('0th', 'sparsity'), 'func': gp.plot_early_stopping},
        {'name': 'final_positive_percent', 'x': ('0th', 'sparsity'), 'func': gp.plot_sign_proportion},
        {'name': 'initial_positive_percent', 'x': ('0th', 'sparsity'), 'func': gp.plot_sign_proportion},
        {'name': 'final_avg_mag', 'x': ('0th', 'sparsity'), 'func': gp.plot_magnitude},
        {'name': 'initial_avg_mag', 'x': ('0th', 'sparsity'), 'func': gp.plot_magnitude},
        {'name': 'best_val_acc', 'x': ('0th', 'sparsity'), 'func': gp.plot_accuracy,
            'kwargs': {
                'title': 'Best Validation Accuracy at Early Stopping',
            },
        },
        {'name': 'loss_before_training', 'x': ('0th', 'sparsity'), 'func': gp.plot_loss,
            'kwargs': {
                'title': 'Loss Before Training',
            },
        },
        {'name': 'acc_before_training', 'x': ('0th', 'sparsity'), 'func': gp.plot_accuracy,
            'kwargs': {
                'title': 'Accuracy From Masked Weights Before Training',
            },
        },
        {'name': 'layer_pos_percent', 'x': ('0th', 'sparsity'), 'func': lp.plot_layerwise_positive_sign_proportion, 
            'kwargs': {
                'layer_names': results['0th']['layer_names'][0]
            },
        },
        {'name': 'layer_avg_mag', 'x': ('0th', 'sparsity'), 'func': lp.plot_layerwise_average_magnitude, 
             'kwargs': {
                'layer_names': results['0th']['layer_names'][0]
             }
        },
    ] 

    for params in plot_params:
        save_location = os.path.join(root, plots_dir, params['name'])
        e_key, t_key = params['x']
        x = results[e_key][t_key]
        y_mean = results['mean'][params['name']]
        y_std = results['std'][params['name']]
        num_samples = results['num_samples'][params['name']]
        args = [x, num_samples, y_mean, y_std]
        
        # Either we are plotting one line, or comparing masked vs. unmasked (or plotting layerwise)
        masked_key = 'masked_' + params['name']
        if trial_aggregations.get(masked_key) is not None:
            masked_mean = results['mean'][masked_key] 
            masked_std = results['std'][masked_key] 
            masked_samples = results['num_samples'][masked_key]
            args += [masked_mean, masked_std]
        kwargs = params.get('kwargs', {})
        kwargs['save_location'] = save_location
        params['func'](*args, **kwargs)
         

