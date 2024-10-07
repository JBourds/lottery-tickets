"""
layerwise_plots.py

Module for creating plots comparing values across layers
of a NN across levels of sparsity.

Author: Jordan Bourdeau
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

def plot_layerwise_average_magnitude(
    x: np.ndarray,
    num_samples: int,
    layer_means: np.ndarray[np.ndarray],
    layer_std: np.ndarray[np.ndarray],
    layer_names: List[str],
    save_location: str = None,
):
    
    _plot_over_layers(
        x=x,
        layer_means=layer_means,
        layer_std=layer_std,
        layer_names=layer_names,
        num_samples=num_samples,
        title="Average Magnitude by Layer",
        x_label="Percent % Sparsity",
        y_label="Average Magnitude",
        invert_x=True,
    ) 

    plt.gca().xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:0.2f}%'))

    save_plot(save_location)

def plot_layerwise_positive_sign_proportion(
    x: np.ndarray,
    num_samples: int,
    layer_means: np.ndarray[np.ndarray],
    layer_std: np.ndarray[np.ndarray],
    layer_names: List[str],
    save_location: str = None,
):
    
    _plot_over_layers(
        x=x,
        layer_means=layer_means,
        layer_std=layer_std,
        layer_names=layer_names,
        num_samples=num_samples,
        title="Positive Weights by Layer",
        x_label="Percent % Sparsity",
        y_label="Percent % Weights",
        invert_x=True,
    ) 

    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:0.2f}%'))
    plt.gca().xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:0.2f}%'))

    save_plot(save_location)

def _plot_over_layers(
    x: np.ndarray,
    num_samples: int,
    layer_means: np.ndarray[np.ndarray],
    layer_std: np.ndarray[np.ndarray],
    layer_names: List[str],
    title: str = "Title",
    x_label: str = "X",
    y_label: str = "Y",
    caption: str = None,
    invert_x: bool = True,
):
    plt.figure(figsize=(8, 6))
    for index, name in enumerate(layer_names):
        is_last_line = index == len(layer_names) - 1
        bp.plot_aggregated_summary_ci(
            x=x,
            y_mean=layer_means[:, index],
            y_std=layer_std[:, index],
            num_samples=num_samples,
            legend=name,
            show_ci_legend=is_last_line,
            invert_x=is_last_line if invert_x else False
        )
    plt.gca().set_title(title)
    plt.gca().set_ylabel(y_label)
    plt.gca().set_xlabel(x_label)
    plt.gca().legend()
    plt.gca().grid()
    if caption is not None:
        plt.figtext(0.1, 0.1, caption, wrap=True, horizontalalignment='center', fontsize=12)
