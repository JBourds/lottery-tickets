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

# ------------------------- Public Base Plotting Functions -------------------------

def plot_aggregated_summary_ci(
    summary: history.ExperimentSummary,
    get_x: callable,
    aggregate_trials: callable,
    confidence: float = 0.95,
    legend: str = None,
    show_ci_legend: bool = True,
    show_max_point: bool = False,
    show_min_point: bool = False,
):
    """
    Function which plots the aggregated summary for an experiment
    using a defined function to get the x and one to aggregate trials.

    Args:
        summary (history.ExperimentSummary): `ExperimentSummary` object containing information about training.
        get_x (callable): Function which gets called on the summary to get x values.
        aggregate_trials (callable): Function which gets called on the summary to produce a 2D array of some
            aggregated trial values.
        confidence (float, optional): Confidence interval level to plot. Defaults to 0.95.
        legend (str, optional): Legend to plot for value being plotted. Defaults to None.
        show_ci_legend (bool, optional): Flag for whether confidnce interval legend is shown. Defaults to True.
        show_max_point (bool, optional): Flag for whether max value point gets shown. Defaults to False.
        show_min_point (bool, optional): Flag for whether min value point gets shown. Defaults to False.
    """
    
    x: list[float] = get_x(summary)
    aggregated_trial_data: list[list[float]] = t_agg.aggregate_across_trials(summary, aggregate_trials)
    
    _aggregate_and_plot_ci(
        x=x, 
        y_2d=aggregated_trial_data, 
        confidence=confidence, 
        legend=legend, 
        show_ci_legend=show_ci_legend, 
        show_max_point=show_max_point, 
        show_min_point=show_min_point
    )

# ------------------------- Private Helper Functions -------------------------

def _aggregate_and_plot_ci(
    x: list[float],
    y_2d: list[list[float]],
    confidence: float = 0.95,
    legend: str = None,
    show_ci_legend: bool = True,
    show_max_point: bool = False,
    show_min_point: bool = False,
    ):
    """
    Function which creates the base line graph for an experiment summary
    and includes confidence intervals over the y-axis values.

    Args:
        x (list[float]): List of floating point values to plot on x axis.
        y_2d (list[list[float]]): 2D list of floating point values to aggregate.
            Dimensions are `# Experiments, # Trials`.
        aggregate_y (callable): Function for aggregating the values across 2D array
          which will go on the y-axis. Examples would be test/train accuracy, early stopping iteration, etc.
        
        NOTE: See `_plot_line_graph_with_confidence_interval` for other argument information.       
    """
    # Compute y-axis values aggregated over experiments along with their standard deviation
    aggregated_values: np.array = e_agg.mean_over_trials(y_2d)
    aggregated_std: np.array = e_agg.std_over_trials(y_2d)
    
    _plot_line_graph_with_confidence_interval(
        x=x, 
        y=aggregated_values, 
        std_y=aggregated_std, 
        confidence=confidence, 
        legend=legend, 
        show_ci_legend=show_ci_legend, 
        show_max_point=show_max_point, 
        show_min_point=show_min_point,
    )

def _annotate_extreme_points(
    x: np.array, 
    y: np.array, 
    use_max: bool = True,
    ):
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
        xmax: float = x[np.argmax(y)]  # Find the x-coordinate corresponding to ymax
        plt.axvline(x=xmax, color='r', linestyle='--', label=f'Max: ({xmax:.2f}, {ymax:.2f})')
    else:
        ymin: float = np.min(y)
        xmin: float = x[np.argmin(y)]  # Find the x-coordinate corresponding to ymin
        plt.axvline(x=xmin, color='b', linestyle='--', label=f'Min: ({xmin:.2f}, {ymin:.2f})')

def _plot_line_graph_with_confidence_interval(
    x: list[float], 
    y: list[float],
    std_y: np.array,
    confidence: float = 0.95,
    legend: str = None,
    show_ci_legend: bool = True,
    show_max_point: bool = False,
    show_min_point: bool = False,
    ):
    """
    Function which creates a base line graph and confidence intervals given some x, y, and standard deviation for
    each y point.

    Args:
        x (list[float]): Floating point x values to plot.
        y (list[float]): Floating point y values to plot, where each y value corresponds to some aggregation
            over a sample.
        std_y (np.array[float]): Array of floating points values of length N where N is the number of points to plot.
            Each value corresponds to the standard deviation of the sample the y point was averaged over.
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
    """
    
    if confidence >= 1 or confidence < 0:
      raise ValueError('Confidence must be between 0 and 1')
    
    # Make sure these are all Numpy arrays
    std_y: np.array = np.array(std_y)
    
    # Calculate Z-score and standard error to make confidence interval
    z_score: float = norm.ppf((1 + confidence) / 2)
    sample_count: int = len(y)
    confidence_interval: np.array = z_score * std_y  / np.sqrt(sample_count)

    plt.plot(x, y, label=legend)
    plt.gca().invert_xaxis()
    plt.fill_between(
        x,
        y - confidence_interval,
        y + confidence_interval,
        color='gray',
        alpha=0.3,
        label=f'{confidence * 100:.2f}% CI' if show_ci_legend else None,
    )
    if show_max_point:
        _annotate_extreme_points(x, y, use_max=True)
    if show_min_point:
        _annotate_extreme_points(x, y, use_max=False)