from src.metrics.features import *
from src.harness.meta import create_meta
from copy import copy as shallowcopy
from itertools import combinations, product
import json
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.ticker as mtick
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from tensorflow import keras
from typing import Callable, Dict, List, Tuple
import os
import sys
import argparse

# Get root in path
root = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))))
sys.path.append(root)

# Takeaways (no training steps):
#     - Large diminishing marginal returns beyond a 1 layer 16 neuron ReLU network


def plot_feature_arch_subsets(variants: Dict[Tuple[int, int], Dict[str, Dict]]):
    plt.figure(figsize=(15, 12))
    plt.title("Architecture Sweep over Feature Subsets")
    plt.xlabel("Feature Subset")
    plt.ylabel("Max Accuracy (%)")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    subset_mappings = {}
    for variant, feature_subsets in variants.items():
        if not subset_mappings:
            subset_mappings = {subset: n for n, subset in enumerate(
                feature_subsets.keys(), start=1)}
        categories = []
        accuracies = []
        for subset, data in feature_subsets.items():
            x_jitter = np.random.uniform(low=-0.4, high=0.4)
            categories.append(subset_mappings[subset] + x_jitter)
            accuracies.append(max(data["accuracy"]))
        plt.scatter(categories, accuracies, label=variant)
        for x, y in zip(categories, accuracies):
            y_jitter = .0075 if np.random.uniform(
                low=0, high=1) > 0.5 else -.0075
            plt.annotate(variant, (x, y + y_jitter), ha="center")
    # Necessary so we can jitter points
    locations = []
    labels = []
    for label, location in subset_mappings.items():
        labels.append(",\n".join(label.split(",")))
        locations.append(location)
    plt.gca().set_xticks(locations, labels)
    plt.tick_params(axis="x", labelsize=8)
    plt.tight_layout()
    plt.grid()
    plt.legend()
    # Don't care about any below this threshold
    plt.ylim(bottom=0.40)
    plt.savefig("arch_feature_subsets.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature importance testing.")
    includes = ["l_sparsity", "sparsity", "l_rel_size"]
    feature_banks = [
        ["norm_wt10_mag", "norm_wt10_synflow", "wt10_sign"],
        ["norm_wi_mag", "norm_wi_synflow", "wi_sign"],
    ]
    feature_subsets = []
    for bank in feature_banks:
        for choose in range(2, len(bank)):
            for subset in map(list, combinations(bank, choose)):
                feature_subsets.append(subset + includes)
                feature_subsets.append(subset)

    widths = 2 ** np.array([3, 5, 7])
    depths = 2 ** np.array([0, 2, 4])
    hidden_layers = [(depth, width)
                     for width, depth in product(widths, depths)]
    df_path = os.path.join(root, "mnist_weightabase.pkl")
    df = pd.read_pickle(df_path)
    epochs = 2
    batch_size = 256
    # {arch: {subset: {data}}}
    variants = {}
    # for depth, width in hidden_layers:
    #    dim = f"{depth},{width}"
    #    variants[dim] = {}
    #    for subset in feature_subsets:
    #        X, Y = featurize_db(df, subset)
    #        indices = np.random.permutation(X.shape[0])
    #        rand_X = X[indices]
    #        rand_Y = Y[indices]
    #        model = create_meta(rand_X[0].shape, depth, width)
    #        history = model.fit(rand_X, rand_Y, epochs=epochs, batch_size=batch_size, validation_split=0.2, shuffle=True)
    #        variants[dim][",".join(subset)] = {
    #            "epochs": epochs,
    #            "batch_size": batch_size,
    #            "accuracy": history.history["accuracy"],
    #        }

    # with open("arch_feature_gridsearch.json", "w") as outfile:
    #     json.dump(variants, outfile)

    with open("arch_feature_gridsearch.json", "r") as infile:
        variants = json.load(infile)

    plot_feature_arch_subsets(variants)
