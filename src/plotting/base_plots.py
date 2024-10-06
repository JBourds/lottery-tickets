"""
base_plots.py

Module containing the basic functions used to generate specific plots.

Author: Jordan Bourdeau
Date Created: 5/2/24
"""

import copy
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from typing import Generator, List

from src.harness import history
from src.metrics import experiment_aggregations as e_agg
from src.metrics import trial_aggregations as t_agg

# ------------------------- Public Base Plotting Functions -------------------------

def plot_aggregated_summary_ci(
    x: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    num_samples: int,
    confidence: float = 0.95,
    legend: str = None,
    show_ci_legend: bool = True,
    show_max_point: bool = False,
    show_min_point: bool = False,
    invert_x: bool = True,
):
    """
    Function which creates a base line graph and confidence intervals given some x, y, and standard deviation for
    each y point.

    Args:
        x (List[float]): Floating point x values to plot.
        y_mean (List[float]): Floating point y values to plot, where each y value corresponds to some aggregation
            over a sample. Dimensions are `# Trials Per Experiment, # Experimnents`.
        y_std (np.array[float]): Array of floating points values of length N where N is the number of points to plot.
            Each value corresponds to the standard deviation of the sample the y point was averaged over.
        num_samples (int): Integer for the number of samples. Used in confidence interval calculation.
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
    y_std = np.array(y_std)
    y_mean = np.array(y_mean)
    x = np.array(x)
    
    # Calculate Z-score and standard error to make confidence interval
    z_score = norm.ppf((1 + confidence) / 2)
    confidence_interval: np.array = z_score * y_std  / np.sqrt(num_samples)
    
    plt.plot(x, y_mean, label=legend)
    if invert_x:
        plt.gca().invert_xaxis()
    
    if show_max_point:
        _annotate_extreme_points(x, y_mean, use_max=True)
    if show_min_point:
        _annotate_extreme_points(x, y_mean, use_max=False)
        
    plt.fill_between(
        x,
        y_mean - confidence_interval,
        y_mean + confidence_interval,
        alpha=0.3,
        label=f'{confidence * 100:.2f}% CI' if show_ci_legend else None,
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

