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
    summary: history.ExperimentSummary, 
    save_location: str = None,
    ):
    """
    Function which plots the best training/test at the point at which
    early stopping occurs.

    Args:
        summary (history.ExperimentSummary): Object containing experiments.
        save_location (str, optional): String to save plot to if it is not None.
    """
    
    # Early Stopping
    plt.figure(figsize=(8,6))
    bp.plot_aggregated_summary_ci(
        summary=summary, 
        get_x=e_agg.get_sparsities,
        aggregate_trials=t_agg.get_early_stopping_iteration, 
        legend='Early Stopping Iteration',
        show_min_point=True,
    )
    plt.gca().set_title(f'Early Stopping Iteration Over Iterative Pruning')
    plt.gca().set_ylabel('Early Stopping Iteration')
    plt.gca().set_xlabel('Sparsity (% Unpruned Weights)')
    plt.gca().legend()
    plt.gca().grid()
    caption: str = f'Each iteration corresponds to a batch of size {C.BATCH_SIZE}'
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=12)
    save_plot(save_location)
    plt.show()

def plot_best_accuracy_at_early_stopping(
    summary: history.ExperimentSummary, 
    train: bool = False,
    save_location: str = None,
    ):
    """
    Function which plots the best training/test at the point at which
    early stopping occurs.

    Args:
        summary (history.ExperimentSummary): Object containing experiments.
        train (bool, optional): Boolean flag for whether to use train or test accuracy. 
            Defaults to False.
        save_location (str, optional): String to save plot to if it is not None.
    """
    best_accuracy: callable = functools.partial(t_agg.get_best_accuracy_percent, train=train)
    accuracy_status: str = 'Train' if train else 'Test'
    
    # Best Accuracy at Early Stopping
    plt.figure(figsize=(8,6))
    bp.plot_aggregated_summary_ci(
        summary=summary, 
        get_x=e_agg.get_sparsities,
        aggregate_trials=best_accuracy, 
        legend='Test Accuracy',
        show_max_point=True,
    )
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:0.2f}%'))
    plt.gca().set_title(f'Best {accuracy_status} Accuracy at Early Stopping Point Over Iterative Pruning')
    plt.gca().set_ylabel('Accuracy %')
    plt.gca().set_xlabel('Sparsity (% Unpruned Weights)')
    plt.gca().legend()
    plt.gca().grid()
    save_plot(save_location)
    plt.show()
    
def plot_sign_proportion(
    summary: history.ExperimentSummary, 
    use_initial_weights: bool = False, 
    find_positive_proportion: bool = True,
    save_location: str = None,
    ):
    """
    Plot the percentage of positive or negative masked weights over iterative pruning.

    Args:
        summary (history.ExperimentData): Experiment data.
        use_initial_weights (bool, optional): Flag for using initial weights. Defaults to False.
        find_positive_proportion (bool, optional): Flag for finding positive or negative proportion. 
            Defaults to True (positive proportion).
        save_location (str, optional): String to save plot to if it is not None.
    """
    sign_function: callable = t_agg.get_global_percent_positive_weights if find_positive_proportion else t_agg.get_global_percent_negative_weights
    proportion_of_weights_function: callable = functools.partial(sign_function, use_initial_weights=use_initial_weights)
    
    sign: str = 'Positive' if find_positive_proportion else 'Negative'
    mask_status: str = 'Initial' if use_initial_weights else 'Final'
    
    # Global Initial Weight Proportion Positive Weights
    plt.figure(figsize=(8,6))
    bp.plot_aggregated_summary_ci(
        summary=summary, 
        get_x=e_agg.get_sparsities,
        aggregate_trials=proportion_of_weights_function,
    )
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:0.2f}%'))
    plt.gca().set_title(f'Percentage of {sign} Masked {mask_status} Weights Over Iterative Pruning')
    plt.gca().set_ylabel(f'Proportion of {sign} Weights')
    plt.gca().set_xlabel('Sparsity (% Unpruned Weights)')
    plt.gca().legend()
    plt.gca().grid()
    save_plot(save_location)
    plt.show()
    
def plot_average_magnitude(
    summary: history.ExperimentSummary, 
    use_initial_weights: bool = False,
    plot_both: bool = False, 
    save_location: str = None,
    ):
    """
    Plot the average magnitude of masked weights over iterative pruning.

    Args:
        summary (history.ExperimentSummary): Experiment data.
        use_initial_weights (bool, optional): Flag for using initial weights. Defaults to False.
        plot_both (bool, optional): Flag for plotting both sets of weights. Defaults to False.
        save_location (str, optional): String to save plot to if it is not None.
    """
    
    magnitude_function: callable = functools.partial(t_agg.get_global_average_magnitude, use_initial_weights=use_initial_weights)
    mask_status: str = 'Initial' if use_initial_weights else 'Final'
    
    # Global Initial Weight Proportion Positive Weights
    plt.figure(figsize=(8,6))
    bp.plot_aggregated_summary_ci(
        summary=summary, 
        get_x=e_agg.get_sparsities,
        aggregate_trials=magnitude_function, 
        legend=f'{mask_status} Weights', 
        show_ci_legend=not plot_both),
    if plot_both:
        second_mask_status: str = 'Final' if use_initial_weights else 'Initial'
        second_magnitude_function: callable = functools.partial(t_agg.get_global_average_magnitude, use_initial_weights=not use_initial_weights)
        bp.plot_aggregated_summary_ci(
            summary=summary, 
            get_x=e_agg.get_sparsities,
            aggregate_trials=second_magnitude_function, 
            legend=f'{second_mask_status} Weights'
        )
        # If we plot this twice, it inverts x axis twice
        plt.gca().invert_xaxis()

    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:0.2f}'))
    plt.gca().set_title(f'Average Magnitude of Masked {mask_status} Weights Over Iterative Pruning')
    plt.gca().set_ylabel('Average Magnitude')
    plt.gca().set_xlabel('Sparsity (% Unpruned Weights)')
    plt.gca().legend()
    plt.gca().grid()
    save_plot(save_location)
    plt.show()
    
def plot_weight_density(
    summary: history.ExperimentSummary, 
    use_initial_weights: bool = False,
    plot_both: bool = False, 
    save_location: str = None,
    ):
    """
    Plot the density histograms of weights across models.

    Args:
        summary (history.ExperimentSummary): Experiment data.
        use_initial_weights (bool, optional): Flag for using initial weights. Defaults to False.
        plot_both (bool, optional): Flag for plotting both sets of weights. Defaults to False.
        save_location (str, optional): String to save plot to if it is not None.
    """
    
    magnitude_function: callable = functools.partial(t_agg.get_global_average_magnitude, use_initial_weights=use_initial_weights)
    mask_status: str = 'Initial' if use_initial_weights else 'Final'
    
    # Global Initial Weight Proportion Positive Weights
    plt.figure(figsize=(8,6))
    bp.plot_aggregated_summary_ci(
        summary=summary, 
        get_x=e_agg.get_sparsities,
        aggregate_trials=magnitude_function, 
        legend=f'{mask_status} Weights', 
        show_ci_legend=not plot_both,
    )
    if plot_both:
        second_mask_status: str = 'Final' if use_initial_weights else 'Initial'
        second_magnitude_function: callable = functools.partial(t_agg.get_global_average_magnitude, use_initial_weights=not use_initial_weights)
        bp.plot_aggregated_summary_ci(
            summary=summary, 
            get_x=e_agg.get_sparsities,
            aggregate_trials=second_magnitude_function, 
            legend=f'{second_mask_status} Weights',
        )

    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:0.2f}'))
    plt.gca().set_title(f'Average Magnitude of Masked {mask_status} Weights Over Iterative Pruning')
    plt.gca().set_ylabel('Average Magnitude')
    plt.gca().set_xlabel('Sparsity (% Unpruned Weights)')
    plt.gca().legend()
    plt.gca().grid()
    save_plot(save_location)
    plt.show()
    
def plot_loss_before_training(
    summary: history.ExperimentSummary,
    save_location: str = None,
    ):
    """
    Plot the loss from reset or masked initial weights.

    Args:
        summary (history.ExperimentSummary): Experiment summary containing experiment data.
        save_location (str, optional): String to save plot to if it is not None.
    """
    loss_before_training: callable = t_agg.get_loss_before_training

    # Global Initial Weight Proportion Positive Weights
    plt.figure(figsize=(8,6))
    bp.plot_aggregated_summary_ci(
        summary=summary, 
        get_x=e_agg.get_sparsities,
        aggregate_trials=loss_before_training, 
        show_min_point=True,
    )
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:0.2f}'))
    plt.gca().set_title('Untrained and Masked Initial Weights Loss')
    plt.gca().set_ylabel('Loss')
    plt.gca().set_xlabel('Sparsity (% Unpruned Weights)')
    plt.gca().legend()
    plt.gca().grid()
    save_plot(save_location)
    plt.show()
    
def plot_accuracy_before_training(
    summary: history.ExperimentSummary,
    save_location: str = None,
    ):
    """
    Plot the loss from reset or masked initial weights.

    Args:
        summary (history.ExperimentSummary): Experiment summary containing experiment data.
        save_location (str, optional): String to save plot to if it is not None.
    """
    accuracy_before_training: callable = t_agg.get_accuracy_before_training

    # Global Initial Weight Proportion Positive Weights
    plt.figure(figsize=(8,6))
    bp.plot_aggregated_summary_ci(
        summary=summary, 
        get_x=e_agg.get_sparsities,
        aggregate_trials=accuracy_before_training,
        show_max_point=True,
    )
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:0.2f}%'))
    plt.gca().set_title('Untrained and Masked Initial Weights Accuracy')
    plt.gca().set_ylabel('Accuracy (%)')
    plt.gca().set_xlabel('Sparsity (% Unpruned Weights)')
    plt.gca().legend()
    plt.gca().grid()
    save_plot(save_location)
    plt.show()
    
# ------------------------- Plots for Doing Batch Summaries -------------------------
