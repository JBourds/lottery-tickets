import matplotlib as mpl
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import gridspec
import numpy as np
import os
from pprint import pprint
import pickle
import sys
from typing import List, Optional

sys.path.append(os.path.join(os.environ["HOME"], "lottery-tickets"))

from src.harness.architecture import Architecture
from src.harness import history

directory = os.path.join(os.environ["HOME"], "lottery-tickets/experiments/archive/lenet_mnist_0_seed_5_experiments_1_batches_0.05_default_sparsity_lm_pruning_20241006-214741/models/model0")
directory = os.path.join(os.environ["HOME"], "lottery-tickets/experiments/archive/conv2_cifar_0_seed_5_experiments_1_batches_0.05_default_sparsity_lm_pruning_20241006-214754/models/model0")
output = os.path.join(os.path.dirname(os.path.dirname(directory)), "plots")
os.makedirs(output, exist_ok=True)
start = "trial0"
end = "trial14"

start_path = os.path.join(directory, start, "trial_data.pkl")
end_path = os.path.join(directory, end, "trial_data.pkl")

initial = history.TrialData.load_from(start_path)
final = history.TrialData.load_from(end_path)


def pixel_plot_2d(
    ax: plt.Axes,
    X: np.ndarray,
    name: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    colorbar: bool = True,
) -> None:
    nrows, ncols = X.shape
    if min_value is None:
        min_value = np.min(X)
    if max_value is None:
        max_value = np.max(X)
    ax.set_title(name)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    cax = ax.matshow(X, cmap="viridis", vmin=min_value, vmax=max_value)
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")

    if colorbar:
        cbar = plt.colorbar(cax, ax=ax)
        cbar.set_label("Values")

def pixel_plot_filter(
    ax: plt.Axes, 
    filters: np.ndarray,
    name: str,
    colorbar: bool = True,
) -> None:
    filter_size = filters.shape[0]  # Assuming filters are of size NxN
    num_filters = filters.shape[2]    # Number of filters in the last dimension
    
    # Calculate grid size
    ncols = int(np.ceil(np.sqrt(num_filters)))
    nrows = int(np.ceil(num_filters / ncols))

    # Determine global vmin and vmax for the color mapping
    vmin = np.min(filters)
    vmax = np.max(filters)

    # Create an array to hold the color data
    composite_array = np.zeros((nrows * filter_size, ncols * filter_size))

    for i in range(num_filters):
        row = i // ncols
        col = i % ncols
        
        # Define position for the current filter
        start_row = row * filter_size
        start_col = col * filter_size

        # Place the filter in the composite array
        composite_array[start_row:start_row + filter_size, start_col:start_col + filter_size] = filters[:, :, i]

    # Plot the composite array
    im = ax.imshow(composite_array, cmap='viridis', vmin=vmin, vmax=vmax)


    # Draw red borders around the filters using vlines and hlines
    for i in range(num_filters):
        row = i // ncols
        col = i % ncols

        start_row = row * filter_size
        start_col = col * filter_size

        # Adjust positions for border lines
        ax.hlines(start_row - 0.5, start_col - 0.5, start_col + filter_size - 0.5, colors='red', linewidth=0.5)  # Top border
        ax.hlines(start_row + filter_size - 0.5, start_col - 0.5, start_col + filter_size - 0.5, colors='red', linewidth=0.5)  # Bottom border
        ax.vlines(start_col - 0.5, start_row - 0.5, start_row + filter_size - 0.5, colors='red', linewidth=0.5)  # Left border
        ax.vlines(start_col + filter_size - 0.5, start_row - 0.5, start_row + filter_size - 0.5, colors='red', linewidth=0.5)  # Right border


    if colorbar:
        cbar = plt.colorbar(im, ax=ax, orientation='vertical')
        cbar.set_label('Filter Values')

    ax.set_title(name)
    ax.axis("off")

def create_pixel_plot(layer: np.ndarray, name: str, suptitle: str = "") -> plt.Figure:
    fig = plt.figure(figsize=(6, 8), constrained_layout=True)
    fig.suptitle(suptitle)

    search = name.lower()
    channel = 0
    name = f"Channel {channel} {name}"

    if len(layer.shape) >= 3:
        layer = layer[:, :, channel, :]
    
    if "dense" in search or "output" in search:
        pixel_plot_2d(fig.gca(), layer, name)
    elif "conv" in search:
        pixel_plot_filter(fig.gca(), layer, name)
    else:
        raise ValueError("Unsupported layer")
    
    return fig

for trial, data in [("Trial 0", initial), ("Trial 14", final)]:
    arch = data.architecture
    layer_names = Architecture.get_model_layers("conv2") 
    weight_layer_indices = [index for index, name in enumerate(layer_names) if "bias" not in name.lower()]
    layer_names = [layer_names[index] for index in weight_layer_indices]
    masks = [data.masks[index] for index in weight_layer_indices]
    initial_weights = [data.initial_weights[index] for index in weight_layer_indices]
    final_weights = [data.initial_weights[index] for index in weight_layer_indices]
    
    label = f"{trial} Masks"
    for weights, name in zip(masks, layer_names):
        fig = create_pixel_plot(weights, name, label)
        fig.savefig(f"{label}-{name}")

    label = f"{trial} Weights"
    for weights, name in zip(final_weights, layer_names):
        fig = create_pixel_plot(weights, name, label)
        fig.savefig(f"{label}-{name}")
    
