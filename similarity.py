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
merged_df = pd.read_pickle(df_path(steps))
keys = ["e_num", "t_num", "l_num"]
merged_df.sort_values(["w_num"], inplace=True)


def model_cos_similarity(weights_0: List[npt.NDArray], weights_1: List[npt.NDArray]) -> List[List[float]]:
    similarities = {"mean": [], "values": []}
    for w0, w1 in zip(weights_0, weights_1):
        if len(w0.shape) == 1:
            w0 = np.reshape(w0.ravel(), (-1, 1))
            w1 = np.reshape(w1.ravel(), (-1, 1))
        sim = cosine_similarity(w0, w1).ravel()
        similarities["mean"].append(np.mean(np.abs(sim)))
        similarities["values"].append(sim)
    return similarities


X_train, X_test, Y_train, Y_test = a.load_data()
np.random.seed(0)
m1 = a.get_model_constructor()()
np.random.seed(1)
m2 = a.get_model_constructor()()

untrained_similarities = model_cos_similarity(
    m1.get_weights(), m2.get_weights())
pprint(untrained_similarities)

masked_m1 = [w * (np.random.uniform(size=w.shape) < .95)
             for w in m1.get_weights()]
masked_m2 = [w * (np.random.uniform(size=w.shape) < .95)
             for w in m2.get_weights()]
masked_similarities = model_cos_similarity(
    masked_m1, masked_m2)
pprint(masked_similarities)

m1.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy())
m1.fit(X_train, Y_train, epochs=5)
m2.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy())
m2.fit(X_train, Y_train, epochs=5)
trained_m1 = m1
trained_m2 = m2

trained_similarities = model_cos_similarity(m1.get_weights(), m2.get_weights())
pprint(trained_similarities)

epath = "/users/j/b/jbourde2/lottery-tickets/11-04-2024/lenet_mnist_0_seed_5_experiments_1_batches_0.025_default_sparsity_lm_pruning_20241102-111614"
experiments = hist.get_experiments(epath)
e0, e1, e2, e3, e4 = experiments
similarities = []
for index, (t0, t1) in enumerate(zip(e0, e1)):
    t0.seed_weights = lambda x: x
    t1.seed_weights = lambda x: x
    t0_weights = [w * m for w, m in zip(t0.initial_weights, t0.masks)]
    t1_weights = [w * m for w, m in zip(t1.initial_weights, t1.masks)]
    similarities.append(model_cos_similarity(t0_weights, t1_weights))

# Build up plot DF
data_dict = defaultdict(list)
for step_num, step in enumerate(similarities):
    # Remove bias layers
    for layer, values in enumerate(step["values"][::2]):
        data_dict["layer"].extend([layer] * len(values))
        data_dict["step"].extend([step_num] * len(values))
        data_dict["values"].extend(values)
df = pd.DataFrame(data_dict)

plot_df = df[(df.step == 0) | (df.step == df.step.max())]
fig, layers = plt.subplots(
    nrows=len(m1.get_weights()) // 2, sharex=True, figsize=(8, 6))
fig.suptitle("Last Layer Cosine Similarity")
for layer, ax in enumerate(layers):
    layer_df = plot_df[plot_df.layer == layer]
    sns.histplot(data=layer_df, x="values", hue="step",
                 ax=ax)
    ax.get_legend().set_title("Pruning Step")
    ax.set_title(f"Layer {layer}")
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


# Are the flattened, masked arrays similar to each other?
for key1, group1 in merged_df.groupby(keys):
    for key2, group2 in merged_df.groupby(keys):
        if should_skip(key1, key2):
            continue
        print(key1, key2)
        x = np.reshape(group1["wi_val"] * group1["w_mask"], -1)
        y = np.reshape(group2["wi_val"] * group2["w_mask"], -1)
        pairwise_similarites[(key1, key2)] = cosine_similarity(x, y)

with open("pairwise_similarities.json", "w") as outfile:
    json.dump(pairwise_similarites, outfile)
