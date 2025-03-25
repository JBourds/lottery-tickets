from src.plotting import base_plots as bp
import argparse
from cka import cca_core
import math
from keras.models import Model
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


def get_activations_model(a: arch.Architecture) -> Model:
    m = a.get_model_constructor()()
    activation_model = Model(inputs=m.inputs, outputs=[
                             l.output for l in m.layers])
    return activation_model


def get_cca(epath: str, inputs: npt.NDArray, a: arch.Architecture, use_initial: bool = True) -> Tuple[npt.NDArray, npt.NDArray]:
    experiments = hist.get_experiments(epath)
    trials = list(map(list, experiments))
    for t in trials:
        for x in t:
            x.seed_weights = lambda x: x
    activations = get_activations_model(a)
    # Pairwise comparison of CCAs between every model as they get sparser
    all_ccas = []
    for index, (trials_0, trials_1) in enumerate(combinations(trials, 2)):
        print(
            f"Combination {index + 1} / {math.factorial(len(trials)) / (2 * math.factorial(len(trials) - 2))}")
        ccas = []
        for index, (t0, t1) in enumerate(zip(trials_0, trials_1)):
            print(f"Trial {index + 1}")
            t0_weights = [
                w * m for w, m in zip(t0.initial_weights, t0.masks)
            ] if use_initial else t0.final_weights
            activations.set_weights(t0_weights)
            t0_outputs = [np.squeeze(t.numpy(), axis=1)
                          for t in activations(inputs)]

            t1_weights = [
                w * m for w, m in zip(t1.initial_weights, t1.masks)
            ] if use_initial else t1.final_weights
            activations.set_weights(t1_weights)
            t1_outputs = [np.squeeze(t.numpy(), axis=1)
                          for t in activations(inputs)]
            res = [np.mean(cca_core.get_cca_similarity(
                m0_act.T, m1_act.T, epsilon=1e-10, verbose=False)["cca_coef1"]) for m0_act, m1_act in zip(t0_outputs, t1_outputs)]
            ccas.append(res)
        all_ccas.append(ccas)
    # Shape: (n_combinations, n_steps, n_layers)
    return np.array(all_ccas)

# TODO: Plot two identical topologies with different datasets


def plot_ccas(
    init_ccas: npt.NDArray,
    final_ccas: npt.NDArray,
    output: str,
    a: arch.Architecture,
):
    fig, (initial_ax, final_ax) = plt.subplots(ncols=2)
    fig.suptitle("CCA Score Over Pruning")

    def plot_cca(ax, cca: npt.NDArray, final: bool):
        mean_ccas = np.mean(cca, axis=0)
        std_ccas = np.std(cca, axis=0)
        # TODO: Make this more robust depending on architecture
        sparsities = 0.8 ** np.arange(mean_ccas.shape[0]) * 100
        ax.set_title("Final Weights" if final else "Initial Weights")
        plt.sca(ax)
        for layer_num in range(mean_ccas.shape[1]):
            bp.plot_aggregated_summary_ci(
                sparsities,
                mean_ccas[:, layer_num],
                std_ccas[:, layer_num],
                cca.shape[0],
                legend=f"Layer {layer_num}",
                show_ci_legend=False,
                invert_x=False,
            )
        ax.invert_xaxis()
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_xlabel("Weights Remaining")
        ax.set_ylabel("CCA Score")
        ax.grid()
        ax.legend()
    plot_cca(initial_ax, init_ccas, False)
    plot_cca(final_ax, final_ccas, True)
    fig.tight_layout()
    fig.savefig(output)


epath = "/users/j/b/jbourde2/lottery-tickets/11-04-2024/lenet_mnist_0_seed_5_experiments_1_batches_0.025_default_sparsity_lm_pruning_20241102-111614"
architecture = "lenet"
dataset = "mnist"
a = arch.Architecture(architecture, dataset)
X_train, X_test, Y_train, Y_test = a.load_data()
X_train, X_test, Y_train, Y_test = train_test_split(np.concatenate(
    (X_train, X_test), axis=0), np.concatenate((Y_train, Y_test), axis=0), test_size=0.2, random_state=42)

inputs = X_test[:1000]
init_ccas = get_cca(epath, inputs, a, use_initial=True)
final_ccas = get_cca(epath, inputs, a, use_initial=False)

plot_ccas(init_ccas, final_ccas,
          f"cca_{architecture}_{dataset}.png", a)

reload(arch)

if __name__ == "__main__":
    epath = "/users/j/b/jbourde2/lottery-tickets/11-04-2024/lenet_mnist_0_seed_5_experiments_1_batches_0.025_default_sparsity_lm_pruning_20241102-111614"
    architecture = "lenet"
    dataset = "mnist"
    parser = argparse.ArgumentParser(
        prog="CCA.py", description="Calculate Centered Kernel Alignment over experiments.")
    parser.add_argument("-e", "--epath", default=epath)
    parser.add_argument("-d", "--dataset", default="mnist")
    parser.add_argument("-a", "--arch", default="lenet")
    args = parser.parse_args()

    a = arch.Architecture(args.arch, args.dataset)
    X_train, X_test, Y_train, Y_test = a.load_data()
    X_train, X_test, Y_train, Y_test = train_test_split(np.concatenate(
        (X_train, X_test), axis=0), np.concatenate((Y_train, Y_test), axis=0), test_size=0.2, random_state=42)

    inputs = X_test[:1000]
    init_linear_ccas = get_cca(args.epath, inputs, a, use_initial=True)
    final_linear_ccas = get_cca(args.epath, inputs, a, use_initial=False)

    plot_ccas(init_linear_ccas, final_linear_ccas,
              f"cca_{args.arch}_{args.dataset}.png", a)
