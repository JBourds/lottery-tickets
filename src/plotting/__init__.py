from matplotlib import pyplot as plt
import os

from src.harness import paths


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
