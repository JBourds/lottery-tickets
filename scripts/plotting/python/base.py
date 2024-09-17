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
    print("Creating plots")
    # Hardcoded to use only the first one for now
    experiments = history.get_experiments(root, models_dir, eprefix, tprefix, tdata)
    aggregations = [
        t_agg.get_best_loss,
        t_agg.get_sparsity_percentage,
        t_agg.get_early_stopping_iteration,
    ]
    best_loss, sparsity_percent, early_stopping_iter = t_agg.aggregate_across_trials(experiments, aggregations)
    print('Best Loss')
    print(best_loss)
    print('Sparsity %')
    print(sparsity_percent)
    print('Early Stopping Iteration')
    print(early_stopping_iter)
    

    
