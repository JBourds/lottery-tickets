from src.plotting import base_plots as bp
import cka.cca_core
import math
from cka.CKA import linear_CKA, kernel_CKA
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


def df_path(steps: int) -> str:
    return os.path.join(
        os.path.expanduser("~"),
        "lottery-tickets",
        "11-04-2024",
        "lenet_mnist_0_seed_5_experiments_1_batches_0.025_default_sparsity_lm_pruning_20241102-111614",
        "data",
        f"merged_{steps}_steps.pkl"
    )


def get_activations_model(a: arch.Architecture) -> Model:
    m = a.get_model_constructor()()
    activation_model = Model(inputs=m.inputs, outputs=[
                             l.output for l in m.layers])
    return activation_model


def get_cka(epath: str, inputs: npt.NDArray, a: arch.Architecture, use_initial: bool = True) -> npt.NDArray:
    experiments = hist.get_experiments(epath)
    trials = list(map(list, experiments))
    for t in trials:
        for x in t:
            x.seed_weights = lambda x: x
    activations = get_activations_model(a)
    # Pairwise comparison of CKAs between every model as they get sparser
    linear_ckas = []
    for index, (trials_0, trials_1) in enumerate(combinations(trials, 2)):
        print(
            f"Combination {index + 1} / {math.factorial(len(trials)) / (2 * math.factorial(len(trials) - 2))}")
        lin_ckas = []
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

            lin_ckas.append([linear_CKA(m0_act, m1_act)
                             for m0_act, m1_act in zip(t0_outputs, t1_outputs)])
        linear_ckas.append(lin_ckas)
    # Shape: (n_combinations, n_steps, n_layers)
    return np.array(linear_ckas)


def plot_ckas(ckas: npt.NDArray, output: str, final_weights: bool):
    mean_ckas = np.mean(ckas, axis=0)
    std_ckas = np.std(ckas, axis=0)
    # TODO: Make this more robust depending on architecture
    sparsities = 0.8 ** np.arange(mean_ckas.shape[0]) * 100
    plt.figure()
    plt.title("CKA Score Over Time " +
              ("(Final Weights)" if final_weights else "(Initial Weights)"))
    for layer_num in range(mean_ckas.shape[1]):
        bp.plot_aggregated_summary_ci(
            sparsities,
            mean_ckas[:, layer_num],
            std_ckas[:, layer_num],
            init_linear_ckas.shape[0],
            legend=f"Layer {layer_num}",
            show_ci_legend=False,
            invert_x=False,
        )
    plt.gca().invert_xaxis()
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())
    plt.legend()
    plt.xlabel("Weights Remaining")
    plt.ylabel("CKA Score")
    plt.savefig(output)


reload(arch)

epath = "/users/j/b/jbourde2/lottery-tickets/11-04-2024/lenet_mnist_0_seed_5_experiments_1_batches_0.025_default_sparsity_lm_pruning_20241102-111614"
architecture = "lenet"
dataset = "mnist"
a = arch.Architecture(architecture, dataset)
X_train, X_test, Y_train, Y_test = a.load_data()


inputs = X_test[:1000]
init_linear_ckas = get_cka(epath, inputs, a, use_initial=True)
plot_ckas(init_linear_ckas, "initial_cka_lenet_mnist.png", final_weights=False)
final_linear_ckas = get_cka(epath, inputs, a, use_initial=False)
plot_ckas(final_linear_ckas, "final_cka_lenet_mnist.png", final_weights=True)
