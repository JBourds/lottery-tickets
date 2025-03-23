import json
import time
from itertools import combinations
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


# ['e_num', 't_num', 'seed', 'step', 'sparsity', 'size', 'l_num', 'l_size',
#  'l_rel_size', 'l_sparsity', 'li_mag_mean', 'li_mag_std',
#  'li_prop_positive', 'lf_mag_mean', 'lf_mag_std', 'lf_prop_positive',
#  'dense', 'bias', 'conv', 'output', 'w_num', 'wi_sign', 'wi_val',
#  'wi_mag', 'wi_perc', 'wi_std', 'wi_synflow', 'wf_sign', 'wf_val',
#  'wf_mag', 'wf_perc', 'wf_std', 'wf_synflow', 'label', 'norm_wi_mag',
#  'norm_wi_synflow', 'wt10_sign', 'wt10_val', 'wt10_mag', 'wt10_perc',
#  'wt10_std', 'wt10_synflow', 'norm_wt10_mag', 'norm_wt10_synflow',
#  'arch_lenet', 'dataset_mnist', 'mag_change', 'norm_mag_change']

reload(meta)
reload(f)
architecture = "lenet"
dataset = "mnist"
a = arch.Architecture(architecture, dataset)
steps = 10
batch_size = 256


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


X_train, X_test, Y_train, Y_test = a.load_data()
np.random.seed(0)
m1 = a.get_model_constructor()()
np.random.seed(1)
m2 = a.get_model_constructor()()

untrained_weight_similarities = model_cos_similarity(
    m1.get_weights(), m2.get_weights())
pprint(untrained_weight_similarities)


def mask_overlap(masks_0: List[npt.NDArray], masks_1: List[npt.NDArray]) -> List[List[float]]:
    overlaps = []
    for m0, m1 in zip(masks_0, masks_1):
        overlap = np.sum(m0 == m1)
        overlaps.append(overlap / m0.size)
    return overlaps


masked_m1 = [w * m for w, m in zip(m1.get_weights(), mask_1)]
masked_m2 = [w * m for w, m in zip(m2.get_weights(), mask_2)]
masked_weight_similarities = model_cos_similarity(
    masked_m1, masked_m2)
pprint(masked_weight_similarities)

m1.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy())
m1.fit(X_train, Y_train, epochs=5)
m2.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy())
m2.fit(X_train, Y_train, epochs=5)
trained_m1 = m1
trained_m2 = m2

trained_weight_similarities = model_cos_similarity(
    m1.get_weights(), m2.get_weights())
pprint(trained_weight_similarities)

epath = "/users/j/b/jbourde2/lottery-tickets/11-04-2024/lenet_mnist_0_seed_5_experiments_1_batches_0.025_default_sparsity_lm_pruning_20241102-111614"
experiments = hist.get_experiments(epath)
trials = list(map(list, experiments))
for t in trials:
    for x in t:
        x.seed_weights = lambda x: x

row_similarities = []
col_similarities = []
for (trials_0, trials_1) in combinations(trials, 2):
    for t0, t1 in zip(trials_0, trials_1):
        t0_weights = [w * m for w, m in zip(t0.initial_weights, t0.masks)]
        t1_weights = [w * m for w, m in zip(t1.initial_weights, t1.masks)]
        row_similarities.append(model_cos_similarity(t0_weights, t1_weights))
        col_similarities.append(model_cos_similarity(
            [w.T for w in t0_weights], [w.T for w in t1_weights]))

# Build up plot DF
row_dict = defaultdict(list)
for step_num, step in enumerate(row_similarities):
    # Remove bias layers
    for layer, values in enumerate(step["values"]):
        row_dict["layer"].extend([layer] * len(values))
        row_dict["step"].extend([step_num] * len(values))
        row_dict["values"].extend(values)
row_df = pd.DataFrame(row_dict)
col_dict = defaultdict(list)
for step_num, step in enumerate(col_similarities):
    # Remove bias layers
    for layer, values in enumerate(step["values"]):
        col_dict["layer"].extend([layer] * len(values))
        col_dict["step"].extend([step_num] * len(values))
        col_dict["values"].extend(values)
col_df = pd.DataFrame(col_dict)

row_plot_df = row_df[(row_df.step == 0) | (row_df.step == row_df.step.max())]
col_plot_df = col_df[(col_df.step == 0) | (col_df.step == col_df.step.max())]
n_layers = len(m1.get_weights()) // 2
fig, axes = plt.subplots(
    nrows=n_layers, sharex=True, ncols=2, figsize=(8, 6))
for layer in range(n_layers):
    # Rows
    row_layer_df = row_plot_df[row_plot_df.layer == layer]
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
    ax = axes[layer, 1]
    sns.histplot(data=col_layer_df, x="values", hue="step",
                 ax=ax)
    ax.get_legend().set_title("Pruning Step")
    ax.set_title(f"Layer {layer} Cols")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Counts")
    ax.set_yscale("log")
fig.tight_layout()
fig.savefig("cosine_similarity.png")


def should_skip(key1: Tuple[int], key2: Tuple[int]) -> bool:
    e1, t1, l1 = key1
    e2, t2, l2 = key2
    duplicate = key1 == key2 or (key1, key2) in pairwise_similarites or (
        key2, key1) in pairwise_similarites
    incompatible = l1 != l2
    return duplicate or incompatible


merged_df = pd.read_pickle(df_path(steps))
keys = ["e_num", "t_num"]
merged_df.sort_values(["w_num"], inplace=True)
layer_dimensions = {layer: shape for layer, shape in enumerate(
    map(np.shape, a.get_model_constructor()().get_weights()))}

# Are the flattened, masked arrays similar to each other?
for key1, group1 in merged_df.groupby(keys):
    for key2, group2 in merged_df.groupby(keys):
        if should_skip(key1, key2):
            continue
        print(key1, key2)
        x = [np.reshape(g["wf_val"], layer_dimensions[l])
             for l, g in group1.groupby("l_num")]
        y = [np.reshape(g["wf_val"], layer_dimensions[l])
             for l, g in group2.groupby("l_num")]
        pairwise_similarites[(key1, key2)] = model_cos_similarity(x, y)
