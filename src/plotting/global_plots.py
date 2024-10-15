"""
global_plots.py

Module containing functions for plotting values from experiments 
across an entire network.

Author: Jordan Bourdeau
Date Created: 5/2/24
"""

import functools
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os
from typing import Generator, List

from src.harness import constants as C
from src.harness import history
from src.harness import paths
from src.metrics import experiment_aggregations as e_agg
from src.metrics import trial_aggregations as t_agg
from src.plotting import base_plots as bp

from . import save_plot
        
# ------------------------- Plots for Single Experiment Summary -------------------------
        
def plot_early_stopping(
    x: np.ndarray,
    num_samples: int,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    save_location: str = None,
):
    plt.figure(figsize=(8, 6))
    bp.plot_aggregated_summary_ci(
        x=x,
        y_mean=y_mean,
        y_std=y_std,
        num_samples=num_samples,
        legend='Early Stopping Iteration',
        show_min_point=True,
    )
    plt.gca().xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:0.2f}%'))
    plt.gca().set_title(f'Early Stopping Iteration Over Iterative Pruning')
    plt.gca().set_ylabel('Early Stopping Iteration')
    plt.gca().set_xlabel('Sparsity (% Unpruned Weights)')
    plt.gca().legend()
    plt.gca().grid()
    caption = f'Each iteration corresponds to a batch of size {C.BATCH_SIZE}'
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=12)
    save_plot(save_location)

def plot_magnitude(
    x: np.ndarray,
    num_samples: int,
    unmasked_y: np.ndarray,
    unmasked_std: np.ndarray,
    masked_y: np.ndarray = None,
    masked_std: np.ndarray = None,
    find_positive_proportion: bool = True,
    save_location: str = None,
):
    """
    Plot to compare the magnitude of the initial values for weights which have been masked off
    against those which have not.
    """
    plt.figure(figsize=(8, 6))
    bp.plot_aggregated_summary_ci(
        x=x,
        y_mean=unmasked_y,
        y_std=unmasked_std,
        num_samples=num_samples,
        legend="Unmasked Weights",
    )
    include_masked = masked_y is not None and masked_std is not None
    if include_masked:
        bp.plot_aggregated_summary_ci(
            x=x,
            y_mean=masked_y,
            y_std=masked_std,
            num_samples=num_samples,
            legend="Masked Weights",
            invert_x=False,
        )
    plt.gca().xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:0.2f}%'))
    plt.gca().set_title(f'Magnitude of' + (' Weights Split by Mask' if include_masked else ' Trained Weights'))
    plt.gca().set_ylabel(f'Magnitude')
    plt.gca().set_xlabel('Sparsity (% Unpruned Weights)')
    plt.gca().legend()
    plt.gca().grid()
    save_plot(save_location)

def plot_sign_proportion(
    x: np.ndarray,
    num_samples: int,
    unmasked_y: np.ndarray,
    unmasked_std: np.ndarray,
    masked_y: np.ndarray = None,
    masked_std: np.ndarray = None,
    find_positive_proportion: bool = True,
    save_location: str = None,
):
    sign: str = 'Positive' if find_positive_proportion else 'Negative'
    
    plt.figure(figsize=(8, 6))
    bp.plot_aggregated_summary_ci(
        x=x,
        y_mean=unmasked_y,
        y_std=unmasked_std,
        num_samples=num_samples,
        legend="Unmasked Weights",
    )
    include_masked = masked_y is not None and masked_std is not None
    if include_masked:
        bp.plot_aggregated_summary_ci(
            x=x,
            y_mean=masked_y,
            y_std=masked_std,
            num_samples=num_samples,
            legend="Masked Weights",
            invert_x=False,
        )
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:0.2f}%'))
    plt.gca().xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:0.2f}%'))
    plt.gca().set_title(f'Percentage of {sign}' + (' Weights Split by Mask' if include_masked else ' Trained Weights'))
    plt.gca().set_ylabel(f'Proportion of {sign} Weights')
    plt.gca().set_xlabel('Sparsity (% Unpruned Weights)')
    plt.gca().legend()
    plt.gca().grid()
    save_plot(save_location)
        
def plot_loss(
    x: np.ndarray,
    num_samples: int,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    title: str = "Loss",
    y_label: str = "Loss",
    x_label: str = "Sparsity (% Unpruned Weights)",
    save_location: str = None,
):
    plt.figure(figsize=(8, 6))
    bp.plot_aggregated_summary_ci(
        x=x,
        y_mean=y_mean,
        y_std=y_std,
        num_samples=num_samples,
        show_min_point=True,
    )
    plt.gca().xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:0.2f}%'))
    plt.gca().set_title(title)
    plt.gca().set_ylabel(y_label)
    plt.gca().set_xlabel(x_label)
    plt.gca().legend()
    plt.gca().grid()
    save_plot(save_location)
        
def plot_accuracy(
    x: np.ndarray,
    num_samples: int,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    title: str = "Unmasked Weight Accuracy",
    y_label: str = "Accuracy (%)",
    x_label: str = "Sparsity (% Unpruned Weights)",
    save_location: str = None,
):
    plt.figure(figsize=(8, 6))
    bp.plot_aggregated_summary_ci(
        x=x,
        y_mean=y_mean,
        y_std=y_std,
        num_samples=num_samples,
        show_max_point=True,
    )
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:0.2f}%'))
    plt.gca().xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:0.2f}%'))
    plt.gca().set_title(title)
    plt.gca().set_ylabel(y_label)
    plt.gca().set_xlabel(x_label)
    plt.gca().legend()
    plt.gca().grid()
    save_plot(save_location)
        
