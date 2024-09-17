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

# ------------------------- Helper Functions -------------------------

def save_plot(location: str):
    """
    Helper function which saves the current plot to a location,
    and creates the path if it does not exist.

    Args:
        location (str): String location to save the plot, or None
            if the plot shouldn't be saved.
    """
    if location:
        directory, _ = os.path.split(location)
        if directory:
            paths.create_path(directory)
        plt.savefig(location)
        
# ------------------------- Plots for Single Experiment Summary -------------------------
        
def plot_early_stopping(
    x: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    num_samples: int,
    save_location: str = None,
):
    # Early Stopping
    plt.figure(figsize=(8,6))
    bp.plot_aggregated_summary_ci(
        x=x,
        y_mean=y_mean,
        y_std=y_std,
        num_samples=num_samples,
        legend='Early Stopping Iteration',
        show_min_point=True,
    )
    plt.gca().set_title(f'Early Stopping Iteration Over Iterative Pruning')
    plt.gca().set_ylabel('Early Stopping Iteration')
    plt.gca().set_xlabel('Sparsity (% Unpruned Weights)')
    plt.gca().legend()
    plt.gca().grid()
    caption = f'Each iteration corresponds to a batch of size {C.BATCH_SIZE}'
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=12)
    save_plot(save_location)
    
def plot_best_accuracy_at_early_stopping(
    x: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    num_samples: int,
    train: bool = False,
    save_location: str = None,
):
    accuracy_status: str = 'Train' if train else 'Test'
    
    # Best Accuracy at Early Stopping
    plt.figure(figsize=(8,6))
    bp.plot_aggregated_summary_ci(
        x=x,
        y_mean=y_mean,
        y_std=y_std,
        num_samples=num_samples,
        legend=f'{accuracy_status} Accuracy',
        show_max_point=True,
    )
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:0.2f}%'))
    plt.gca().set_title(f'Best {accuracy_status} Accuracy at Early Stopping Point Over Iterative Pruning')
    plt.gca().set_ylabel('Accuracy %')
    plt.gca().set_xlabel('Sparsity (% Unpruned Weights)')
    plt.gca().legend()
    plt.gca().grid()
    save_plot(save_location)
        
def plot_sign_proportion(
    x: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    num_samples: int,
    use_initial_weights: bool = False, 
    find_positive_proportion: bool = True,
    save_location: str = None,
):
    sign: str = 'Positive' if find_positive_proportion else 'Negative'
    mask_status: str = 'Initial' if use_initial_weights else 'Final'
    
    # Global Initial Weight Proportion Positive Weights
    plt.figure(figsize=(8,6))
    bp.plot_aggregated_summary_ci(
        x=x,
        y_mean=y_mean,
        y_std=y_std,
        num_samples=num_samples,
    )
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:0.2f}%'))
    plt.gca().set_title(f'Percentage of {sign} Masked {mask_status} Weights Over Iterative Pruning')
    plt.gca().set_ylabel(f'Proportion of {sign} Weights')
    plt.gca().set_xlabel('Sparsity (% Unpruned Weights)')
    plt.gca().legend()
    plt.gca().grid()
    save_plot(save_location)
        
# TODO: Add special handling for this plot
#def plot_average_magnitude(
#    x: np.ndarray,
#    y_mean: np.ndarray,
#    y_std: np.ndarray,
#    num_samples: int,
#    
#    use_initial_weights: bool = False,
#    plot_both: bool = False, 
#    save_location: str = None,
#):
#    magnitude_function: callable = functools.partial(t_agg.get_global_average_magnitude, use_initial_weights=use_initial_weights)
#    mask_status: str = 'Initial' if use_initial_weights else 'Final'
#    
#    # Global Initial Weight Proportion Positive Weights
#    plt.figure(figsize=(8,6))
#    bp.plot_aggregated_summary_ci(
#        experiments=experiments, 
#        get_x=e_agg.get_sparsities,
#        aggregate_trials=magnitude_function, 
#        legend=f'{mask_status} Weights', 
#        show_ci_legend=not plot_both),
#    if plot_both:
#        second_mask_status: str = 'Final' if use_initial_weights else 'Initial'
#        second_magnitude_function: callable = functools.partial(t_agg.get_global_average_magnitude, use_initial_weights=not use_initial_weights)
#        bp.plot_aggregated_summary_ci(
#            experiments=experiments, 
#            get_x=e_agg.get_sparsities,
#            aggregate_trials=second_magnitude_function, 
#            legend=f'{second_mask_status} Weights'
#        )
#        # If we plot this twice, it inverts x axis twice
#        plt.gca().invert_xaxis()
#
#    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:0.2f}'))
#    plt.gca().set_title(f'Average Magnitude of Masked {mask_status} Weights Over Iterative Pruning')
#    plt.gca().set_ylabel('Average Magnitude')
#    plt.gca().set_xlabel('Sparsity (% Unpruned Weights)')
#    plt.gca().legend()
#    plt.gca().grid()
#    save_plot(save_location)
        
def plot_loss_before_training(
    x: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    num_samples: int,
    save_location: str = None,
):
    loss_before_training: callable = t_agg.get_loss_before_training

    # Global Initial Weight Proportion Positive Weights
    plt.figure(figsize=(8,6))
    bp.plot_aggregated_summary_ci(
        x=x,
        y_mean=y_mean,
        y_std=y_std,
        num_samples=num_samples,
        show_min_point=True,
    )
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:0.2f}'))
    plt.gca().set_title('Untrained and Masked Initial Weights Loss')
    plt.gca().set_ylabel('Loss')
    plt.gca().set_xlabel('Sparsity (% Unpruned Weights)')
    plt.gca().legend()
    plt.gca().grid()
    save_plot(save_location)
        
def plot_accuracy_before_training(
    x: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    num_samples: int,
    save_location: str = None,
):
    accuracy_before_training: callable = t_agg.get_accuracy_before_training

    # Global Initial Weight Proportion Positive Weights
    plt.figure(figsize=(8,6))
    bp.plot_aggregated_summary_ci(
        x=x,
        y_mean=y_mean,
        y_std=y_std,
        num_samples=num_samples,
        show_max_point=True,
    )
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:0.2f}%'))
    plt.gca().set_title('Untrained and Masked Initial Weights Accuracy')
    plt.gca().set_ylabel('Accuracy (%)')
    plt.gca().set_xlabel('Sparsity (% Unpruned Weights)')
    plt.gca().legend()
    plt.gca().grid()
    save_plot(save_location)
        
