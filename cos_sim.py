import json
import time
from itertools import chain, combinations
from collections import defaultdict
import seaborn as sns
from src.metrics.synflow import compute_synflow_per_weight
from src.metrics import features as f
from src.harness import history as hist
from src.harness import meta
from src.harness import dataset as ds
from src.harness import architecture as arch
from typing import Dict, Generator, List, Tuple
from tqdm import tqdm
import os
from pprint import pprint
import pandas as pd
import matplotlib.ticker as mtick
from matplotlib import pyplot as plt
import numpy as np
from importlib import reload
import numpy.typing as npt
from tensorflow import keras
from copy import copy as shallowcopy
from copy import deepcopy
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
tf.keras.backend.set_floatx('float64')


def df_path(steps: int) -> str:
    return os.path.join(
        os.path.expanduser("~"),
        "lottery-tickets",
        "11-04-2024",
        "lenet_mnist_0_seed_5_experiments_1_batches_0.025_default_sparsity_lm_pruning_20241102-111614",
        "data",
        f"merged_{steps}_steps.pkl"
    )


def output_path(epath: str, filename: str) -> str:
    output_dir = os.path.join(epath, "plots")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename)


def model_cos_similarity(weights_0: List[npt.NDArray], weights_1: List[npt.NDArray]) -> List[List[float]]:
    weight_similarities = {"mean": [], "values": []}
    for w0, w1 in zip(weights_0, weights_1):
        # Skip biases
        if len(w0.shape) == 1:
            continue
        sim = cosine_similarity(w0, w1).ravel()
        weight_similarities["mean"].append(np.mean(np.abs(sim)))
        weight_similarities["values"].append(sim)
    return weight_similarities


def diff_datasets_cos_sim(epath1: str, epath2: str, output: str, nlayers: int, use_initial: bool = True):
    exp1 = hist.get_experiments(epath1)
    exp2 = hist.get_experiments(epath2)
    trials1 = list(map(list, exp1))
    trials2 = list(map(list, exp2))
    pairings = []
    # Create all groupings of a set of trials here
    for t1 in trials1:
        for t2 in trials2:
            pairings.append((t1, t2))

    for t in chain(trials1, trials2):
        for x in t:
            x.seed_weights = lambda x: x

    row_similarities = defaultdict(list)
    col_similarities = defaultdict(list)
    # Compare a single trial for now
    for index, (t0, t1) in enumerate(pairings):
        if use_initial:
            t0_weights = [w * m for w, m in zip(t0.initial_weights, t0.masks)]
            t1_weights = [w * m for w, m in zip(t1.initial_weights, t1.masks)]
        else:
            t0_weights = [w * m for w, m in zip(t0.final_weights, t0.masks)]
            t1_weights = [w * m for w, m in zip(t1.final_weights, t1.masks)]
        row_similarities[index].append(
            model_cos_similarity(t0_weights, t1_weights))
        col_similarities[index].append(model_cos_similarity(
            [w.T for w in t0_weights], [w.T for w in t1_weights]))

    def make_df(similarities: Dict[int, List]) -> pd.DataFrame:
        data_dict = defaultdict(list)
        for step_num, steps in similarities.items():
            for step in steps:
                for layer, values in enumerate(step["values"]):
                    data_dict["layer"].extend([layer] * len(values))
                    data_dict["step"].extend([step_num] * len(values))
                    data_dict["values"].extend(values)
        df = pd.DataFrame(data_dict)
        return df

    row_df = make_df(row_similarities)
    col_df = make_df(col_similarities)

    row_plot_df = row_df[(row_df.step == 0) | (
        row_df.step == row_df.step.max())]
    col_plot_df = col_df[(col_df.step == 0) | (
        col_df.step == col_df.step.max())]
    fig, axes = plt.subplots(
        nrows=nlayers, sharex=True, ncols=2, figsize=(8, 6))
    for layer in range(nlayers):
        # Rows
        row_layer_df = row_plot_df[row_plot_df.layer == layer]
        print("Row Layer Shape:", row_layer_df.shape)
        ax = axes[layer, 0]
        sns.histplot(data=row_layer_df, x="values", hue="step",
                     ax=ax)
        ax.get_legend().set_title("Pruning Step")
        ax.set_title(f"Layer {layer} Rows")
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Counts")
        ax.set_yscale("log")
        # Columns
        col_layer_df = col_plot_df[col_plot_df.layer == layer]
        print("Col Layer Shape:", col_layer_df.shape)
        ax = axes[layer, 1]
        sns.histplot(data=col_layer_df, x="values", hue="step",
                     ax=ax)
        ax.get_legend().set_title("Pruning Step")
        ax.set_title(f"Layer {layer} Cols")
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Counts")
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig("diff_" + output)


def plot_cosine_similarities(epath: str, output: str, nlayers: int, use_initial: bool = True):
    experiments = hist.get_experiments(epath)
    trials = list(map(list, experiments))
    for t in trials:
        for x in t:
            x.seed_weights = lambda x: x

    row_similarities = defaultdict(list)
    col_similarities = defaultdict(list)
    for (trials_0, trials_1) in combinations(trials, 2):
        for index, (t0, t1) in enumerate(zip(trials_0, trials_1)):
            if use_initial:
                t0_weights = [w * m for w,
                              m in zip(t0.initial_weights, t0.masks)]
                t1_weights = [w * m for w,
                              m in zip(t1.initial_weights, t1.masks)]
            else:
                t0_weights = [w * m for w,
                              m in zip(t0.final_weights, t0.masks)]
                t1_weights = [w * m for w,
                              m in zip(t1.final_weights, t1.masks)]
            row_similarities[index].append(
                model_cos_similarity(t0_weights, t1_weights))
            col_similarities[index].append(model_cos_similarity(
                [w.T for w in t0_weights], [w.T for w in t1_weights]))

    def make_df(similarities: Dict[int, List]) -> pd.DataFrame:
        data_dict = defaultdict(list)
        for step_num, steps in similarities.items():
            for step in steps:
                for layer, values in enumerate(step["values"]):
                    data_dict["layer"].extend([layer] * len(values))
                    data_dict["step"].extend([step_num] * len(values))
                    data_dict["values"].extend(values)
        df = pd.DataFrame(data_dict)
        return df

    row_df = make_df(row_similarities)
    col_df = make_df(col_similarities)

    row_plot_df = row_df[(row_df.step == 0) | (
        row_df.step == row_df.step.max())]
    col_plot_df = col_df[(col_df.step == 0) | (
        col_df.step == col_df.step.max())]
    fig, axes = plt.subplots(
        nrows=nlayers, sharex=True, ncols=2, figsize=(8, 6))
    for layer in range(nlayers):
        # Rows
        row_layer_df = row_plot_df[row_plot_df.layer == layer]
        print("Row Layer Shape:", row_layer_df.shape)
        ax = axes[layer, 0]
        sns.histplot(data=row_layer_df, x="values", hue="step",
                     ax=ax)
        ax.get_legend().set_title("Pruning Step")
        ax.set_title(f"Layer {layer} Rows")
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Counts")
        ax.set_yscale("log")
        # Columns
        col_layer_df = col_plot_df[col_plot_df.layer == layer]
        print("Col Layer Shape:", col_layer_df.shape)
        ax = axes[layer, 1]
        sns.histplot(data=col_layer_df, x="values", hue="step",
                     ax=ax)
        ax.get_legend().set_title("Pruning Step")
        ax.set_title(f"Layer {layer} Cols")
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Counts")
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(output)


if __name__ == "__main__":
    epath = "/users/j/b/jbourde2/lottery-tickets/11-04-2024/lenet_mnist_0_seed_5_experiments_1_batches_0.025_default_sparsity_lm_pruning_20241102-111614"
    epath2 = "/users/j/b/jbourde2/lottery-tickets/11-04-2024/lenet_fashion_mnist_0_seed_5_experiments_1_batches_0.025_default_sparsity_lm_pruning_20241102-111614"
    architecture = "lenet"
    dataset = "mnist"
    a = arch.Architecture(architecture, dataset)
    nlayers = len(arch.Architecture.get_model_layers(architecture)) // 2
    steps = 10
    batch_size = 256
    plot_cosine_similarities(
        epath, "lenet_initial_cosine_similarity.png", nlayers, use_initial=True)
    plot_cosine_similarities(
        epath, "lenet_final_cosine_similarity.png", nlayers, use_initial=False)

    diff_datasets_cos_sim(
        epath, epath2, "lenet_initial_cosine_similarity.png", nlayers, use_initial=True)
    diff_datasets_cos_sim(
        epath, epath2, "lenet_final_cosine_similarity.png", nlayers, use_initial=False)

    epath = "/users/j/b/jbourde2/lottery-tickets/11-04-2024/conv2_mnist_0_seed_5_experiments_1_batches_0.025_default_sparsity_lm_pruning_20241102-111614/"
    epath2 = "/users/j/b/jbourde2/lottery-tickets/11-04-2024/conv2_fashion_mnist_0_seed_5_experiments_1_batches_0.025_default_sparsity_lm_pruning_20241102-111614"
    architecture = "conv2"
    dataset = "mnist"
    a = arch.Architecture(architecture, dataset)
    nlayers = len(arch.Architecture.get_model_layers(architecture)) // 2
    steps = 10
    batch_size = 256
    plot_cosine_similarities(
        epath, "conv2_mnist_initial_cosine_similarity.png", nlayers, use_initial=True)
    plot_cosine_similarities(
        epath, "conv2_mnist_final_cosine_similarity.png", nlayers, use_initial=False)

    diff_datasets_cos_sim(
        epath, epath2, "conv2_mnist_initial_cosine_similarity.png", nlayers, use_initial=True)
    diff_datasets_cos_sim(
        epath, epath2, "conv2_mnist_final_cosine_similarity.png", nlayers, use_initial=False)

    epath = "/users/j/b/jbourde2/lottery-tickets/11-04-2024/conv2_cifar_0_seed_5_experiments_1_batches_0.025_default_sparsity_lm_pruning_20241102-111614"
    architecture = "conv2"
    dataset = "cifar10"
    a = arch.Architecture(architecture, dataset)
    nlayers = len(arch.Architecture.get_model_layers(architecture)) // 2
    steps = 10
    batch_size = 256
    plot_cosine_similarities(
        epath, "conv2_cifar_initial_cosine_similarity.png", nlayers, use_initial=True)
    plot_cosine_similarities(
        epath, "conv2_cifar_final_cosine_similarity.png", nlayers, use_initial=False)
