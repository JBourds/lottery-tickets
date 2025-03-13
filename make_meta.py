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


def df_path(steps: int) -> str:
    return os.path.join(os.path.expanduser("~"), "lottery-tickets", "scripts", "meta", "data", f"merged_{steps}_steps.pkl")


def df_features(steps: int) -> List[str]:
    features = [
        "l_rel_size",
        "l_sparsity",
        "li_prop_positive",
        "li_mag_mean",
        "li_mag_std",
        "dense",
        "bias",
        "conv",
        "output",
        "norm_wi_mag",
        "norm_wi_synflow",
    ]
    if steps > 0:
        features.extend([
            f"norm_wt{steps}_mag",
            f"norm_wt{steps}_synflow",
        ])
    return features


merged_df = pd.read_pickle(df_path(1))
merged_df = pd.read_pickle(df_path(5))
merged_df = pd.read_pickle(df_path(10))
pprint(list(merged_df.columns))

reload(meta)
reload(f)
features = df_features(10)
merged_df[features]
architecture = "lenet"
dataset = "mnist"
depth = 1
width = 32
a = arch.Architecture(architecture, dataset)
model = a.get_model_constructor()()
merged_df["mag_change"] = merged_df["wt10_mag"] - merged_df["wi_mag"]
merged_df["norm_mag_change"] = merged_df["mag_change"].transform(
    f.normalize)
X, Y = f.featurize_db(merged_df, features)
meta_model = meta.create_meta(X[0].shape, depth, width)
history, loss, accuracy = meta.train_meta(meta_model, X, Y, epochs=5)
masks, accuracies = meta.make_meta_mask(
    meta_model, meta.make_x, architecture, dataset, 10, features)


# Analyze types of errors made
