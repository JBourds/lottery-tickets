"""
base_plots.py

Module containing the basic functions used to generate specific plots.

Author: Jordan Bourdeau
Date Created: 5/2/24
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


from src.harness import history
from src.metrics import experiment_aggregations as e_agg
from src.metrics import trial_aggregations as t_agg

def annotate_extreme_points(x: np.array, y: np.array, use_max: bool = True):
    """
    Annotates the maximum or minimum points in a plot.

    Args:
        x (list or numpy.ndarray): x-values of the plot.
        y (list or numpy.ndarray): y-values of the plot.
        use_max (bool, optional): Flag for whether to annotate the maximum or 
            minimum point. Defaults to True and annotates the max.
    """
    if use_max:
        ymax: float = np.max(y)
        xmax: float = x[y.index(ymax)]  # Find the x-coordinate corresponding to ymax
        plt.axvline(x=xmax, color='r', linestyle='--', label=f'Max: ({xmax:.2f}, {ymax:.2f})')
    else:
        ymin: float = np.min(y)
        xmin: float = x[y.index(ymin)]  # Find the x-coordinate corresponding to ymin
        plt.axvline(x=xmin, color='b', linestyle='--', label=f'Min: ({xmin:.2f}, {ymin:.2f})')

def create_line_graph_with_confidence_intervals_over_sparsities(
    experiment_summary: history.ExperimentSummary,
    aggregate_experiment_values: callable,
    confidence: float = 0.95,
    legend: str = None,
    show_ci_legend: bool = True,
    show_max_point: bool = False,
    show_min_point: bool = False,
    ) -> plt.figure:
    """
    Function which creates the base line graph for an experiment summary
    and includes confidence intervals over the y-axis values.

    Args:
        experiment_summary (history.ExperimentSummary): Object with data about the experiment.
        aggregate_trial_values (callable): Function used to aggregate values for trials which will
          go on the x-axis. Often will be something like a function retrieving sparsity.
        aggregate_experiment_values (callable): Function for aggregating the values across experiments
          which will go on the y-axis. Examples would be test/train accuracy, early stopping iteration, etc.
        confidence (float, optional): Confidence level to use when plotting confidence intervals.
          Must be between 0 and 1. Defaults to 0.95.
        legend (str, optional): Optional legend to plot the line with.
        show_ci_legend (bool, optional): Optional flag for whether confidence interval legend should be shown.
            Useful for overlaying multiple plots. Defaults to True.
        show_max_point (bool, optional): Optional flag for whether the maximum value point should be shown.
            Defaults to False.
        show_min_point (bool, optional): Optional flag for whether the minimum value point should be shown.
                    Defaults to False.
    Raises:
        ValueError: Confidence interval cannot be >= 1 or < 0.

    Returns:
        None: Shows plot.
    """

    if confidence >= 1 or confidence < 0:
      raise ValueError('Confidence must be between 0 and 1')

    # Compute x-axis values. Will often be something like sparsity %
    sparsities: np.array = experiment_summary.aggregate_across_experiments(t_agg.get_sparsity_percentage, e_agg.mean_over_experiments)
    
    # Compute y-axis values aggregated over experiments along with their standard deviation
    aggregated_experiment_values: np.array = experiment_summary.aggregate_across_experiments(aggregate_experiment_values, e_agg.mean_over_experiments)
    aggregated_experiment_std: np.array = experiment_summary.aggregate_across_experiments(aggregate_experiment_values, e_agg.std_over_experiments)

    # Calculate Z-score and standard error to make confidence interval
    z_score: float = norm.ppf((1 + confidence) / 2)
    aggregated_experiment_std: np.array = np.array(aggregated_experiment_std)
    sample_count: int = len(experiment_summary.experiments)
    confidence_interval: np.array = z_score * aggregated_experiment_std  / np.sqrt(sample_count)

    plt.plot(sparsities, aggregated_experiment_values, label=legend)
    plt.gca().invert_xaxis()
    plt.fill_between(
        sparsities,
        aggregated_experiment_values - confidence_interval,
        aggregated_experiment_values + confidence_interval,
        color='gray',
        alpha=0.3,
        label=f'{confidence * 100:.2f}% CI' if show_ci_legend else None,
    )
    if show_max_point:
        annotate_extreme_points(sparsities, aggregated_experiment_values, use_max=True)
    if show_min_point:
        annotate_extreme_points(sparsities, aggregated_experiment_values, use_max=False)
