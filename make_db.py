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
tf.keras.backend.set_floatx('float64')

# Create the database
epath = "/users/j/b/jbourde2/lottery-tickets/experiments/11-04-2024/lenet_mnist_0_seed_5_experiments_1_batches_0.025_default_sparsity_lm_pruning_20241102-111614"
trials_df, layers_df, weights_df, trained_weights_df = f.build_dataframes(
    epath, train_steps=10, batch_size=64)
trials_df.to_pickle("trials.pkl")
layers_df.to_pickle("layers.pkl")
weights_df.to_pickle("weights.pkl")
trained_weights_df.to_pickle("trained_weights.pkl")
merged_df = f.merge_dfs(trials_df, layers_df, weights_df, trained_weights_df)
merged_df.to_pickle("merged.pkl")

# Read the database


merged_df = pd.read_pickle("merged.pkl")
merged_df.head()
merged_df.label.value_counts()
pprint(list(merged_df.columns))

reload(f)
reload(meta)
# Create a meta model and test it
architecture = "lenet"
dataset = "mnist"
depth = 1
width = 16
a = arch.Architecture(architecture, dataset)
model = a.get_model_constructor()()
features = [
    "l_sparsity",
    "l_rel_size",
    "li_prop_positive",
    "norm_wi_mag",
    "norm_wi_synflow",
]
X, Y = f.featurize_db(merged_df, features)
meta_model = meta.create_meta(X[0].shape, depth, width)
history, loss, accuracy = meta.train_meta(meta_model, X, Y)
masks, accuracies = meta.make_meta_mask(
    meta_model, meta.make_x, architecture, dataset, 50)

# Analyze types of errors made
