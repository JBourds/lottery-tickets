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

from src.harness import history
from src.metrics import trial_aggregations as t_agg
from src.plotting import base_plots as bp


def plot_best_accuracy_at_early_stopping(
    summary: history.ExperimentSummary, 
    train: bool = False,
    ):
    """
    Function which plots the best training/test at the point at which
    early stopping occurs.

    Args:
        summary (history.ExperimentSummary): Object containing experiments.
        train (bool, optional): Boolean flag for whether to use train or test accuracy. 
            Defaults to False.
    """
    best_accuracy: callable = functools.partial(t_agg.get_best_accuracy_percent, train=train)
    accuracy_status: str = 'Train' if train else 'Test'
    
    # Best Accuracy at Early Stopping
    plt.figure()
    bp.create_line_graph_with_confidence_intervals_over_sparsities(summary, best_accuracy, legend='Test Accuracy')
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:0.2f}%'))
    plt.gca().set_title(f'Best {accuracy_status} Accuracy at Early Stopping Point Over Iterative Pruning')
    plt.gca().set_ylabel('Accuracy %')
    plt.gca().set_xlabel('Sparsity (% Unpruned Weights)')
    plt.gca().legend()
    plt.gca().grid()
    plt.show()
    
def plot_sign_proportion(
    summary: history.ExperimentSummary, 
    use_initial_weights: bool = False, 
    find_positive_proportion: bool = True,
    ):
    """
    Plot the percentage of positive or negative masked weights over iterative pruning.

    Args:
        summary (history.ExperimentData): Experiment data.
        use_initial_weights (bool, optional): Flag for using initial weights. Defaults to False.
        find_positive_proportion (bool, optional): Flag for finding positive or negative proportion. 
            Defaults to True (positive proportion).
    """
    sign_function: callable = t_agg.get_global_percent_positive_weights if find_positive_proportion else t_agg.get_global_percent_negative_weights
    proportion_of_weights_function: callable = functools.partial(sign_function, use_initial_weights=use_initial_weights)
    
    sign: str = 'Positive' if find_positive_proportion else 'Negative'
    mask_status: str = 'Initial' if use_initial_weights else 'Final'
    
    # Global Initial Weight Proportion Positive Weights
    plt.figure()
    bp.create_line_graph_with_confidence_intervals_over_sparsities(summary, proportion_of_weights_function)
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:0.2f}%'))
    plt.gca().set_title(f'Percentage of {sign} Masked {mask_status} Weights Over Iterative Pruning')
    plt.gca().set_ylabel(f'Proportion of {sign} Weights')
    plt.gca().set_xlabel('Sparsity (% Unpruned Weights)')
    plt.gca().legend()
    plt.gca().grid()
    plt.show()
    
def plot_average_magnitude(
    summary: history.ExperimentSummary, 
    use_initial_weights: bool = False,
    plot_both: bool = False, 
    ):
    """
    Plot the average magnitude of masked weights over iterative pruning.

    Args:
        summary (history.ExperimentSummary): Experiment data.
        use_initial_weights (bool, optional): Flag for using initial weights. Defaults to False.
        plot_both (bool, optional): Flag for plotting both sets of weights. Defaults to False.
    """
    
    magnitude_function: callable = functools.partial(t_agg.get_global_average_magnitude, use_initial_weights=use_initial_weights)
    mask_status: str = 'Initial' if use_initial_weights else 'Final'
    
    # Global Initial Weight Proportion Positive Weights
    plt.figure()
    bp.create_line_graph_with_confidence_intervals_over_sparsities(summary, magnitude_function, legend=f'{mask_status} Weights', show_ci_legend=not plot_both)
    if plot_both:
        second_mask_status: str = 'Final' if use_initial_weights else 'Initial'
        second_magnitude_function: callable = functools.partial(t_agg.get_global_average_magnitude, use_initial_weights=not use_initial_weights)
        bp.create_line_graph_with_confidence_intervals_over_sparsities(summary, second_magnitude_function, legend=f'{second_mask_status} Weights')

    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:0.2f}'))
    plt.gca().set_title(f'Average Magnitude of Masked {mask_status} Weights Over Iterative Pruning')
    plt.gca().set_ylabel('Average Magnitude')
    plt.gca().set_xlabel('Sparsity (% Unpruned Weights)')
    plt.gca().legend()
    plt.gca().grid()
    plt.show()
    
def plot_loss_before_training(summary: history.ExperimentSummary):
    """
    Plot the loss from reset or masked initial weights.

    Args:
        summary (history.ExperimentSummary): Experiment summary containing experiment data.
    """
    loss_before_training: callable = t_agg.get_loss_before_training

    # Global Initial Weight Proportion Positive Weights
    plt.figure()
    bp.create_line_graph_with_confidence_intervals_over_sparsities(summary, loss_before_training)
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:0.2f}'))
    plt.gca().set_title('Loss from Reset/Masked Initial Weights')
    plt.gca().set_ylabel('Loss')
    plt.gca().set_xlabel('Sparsity (% Unpruned Weights)')
    plt.gca().legend()
    plt.gca().grid()
    plt.show()
    
def plot_accuracy_before_training(summary: history.ExperimentSummary):
    """
    Plot the loss from reset or masked initial weights.

    Args:
        summary (history.ExperimentSummary): Experiment summary containing experiment data.
    """
    accuracy_before_training: callable = t_agg.get_accuracy_before_training

    # Global Initial Weight Proportion Positive Weights
    plt.figure()
    bp.create_line_graph_with_confidence_intervals_over_sparsities(summary, accuracy_before_training)
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:0.2f}%'))
    plt.gca().set_title('Accuracy (%) from Reset/Masked Initial Weights')
    plt.gca().set_ylabel('Accuracy (%)')
    plt.gca().set_xlabel('Sparsity (% Unpruned Weights)')
    plt.gca().legend()
    plt.gca().grid()
    plt.show()