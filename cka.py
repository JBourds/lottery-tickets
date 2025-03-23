import cka.cca_core
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


reload(arch)

epath = "/users/j/b/jbourde2/lottery-tickets/11-04-2024/lenet_mnist_0_seed_5_experiments_1_batches_0.025_default_sparsity_lm_pruning_20241102-111614"
epath2 = "/users/j/b/jbourde2/lottery-tickets/11-04-2024/lenet_fashion_mnist_0_seed_5_experiments_1_batches_0.025_default_sparsity_lm_pruning_20241102-111614"
architecture = "lenet"
dataset = "mnist"
a = arch.Architecture(architecture, dataset)
X_train, X_test, Y_train, Y_test = a.load_data()


def get_activations_model(a: arch.Architecture) -> Model:
    m = a.get_model_constructor()()
    activation_model = Model(inputs=m.inputs, outputs=[
                             l.output for l in m.layers])
    return activation_model


np.random.seed(0)
m0 = get_activations_model(a)
m0_outputs = [np.squeeze(t.numpy(), axis=1) for t in m0(X_train)]

np.random.seed(1)
m1 = get_activations_model(a)
m1_outputs = [np.squeeze(t.numpy(), axis=1) for t in m1(X_train)]

for m0_act, m1_act in zip(m0_outputs, m1_outputs):
    print(
        f"M0 Activation Shape: {m0_act.shape}, M1 Activation Shape: {m1_act.shape}")

linear_ckas = [linear_CKA(m0_act, m1_act)
               for m0_act, m1_act in zip(m0_outputs, m1_outputs)]
linear_ckas
