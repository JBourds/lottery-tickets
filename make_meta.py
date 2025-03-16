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
from sklearn.metrics import accuracy_score
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


def correct_df(df: pd.DataFrame) -> pd.DataFrame:
    df = f.correct_class_imbalance(df[df["keep"] == 1])
    df.drop(columns=["keep"], inplace=True)
    return df


reload(meta)
reload(f)

# Question: Can we iteratively prune using another model to predict whether
# a weight should get pruned?
architecture = "lenet"
dataset = "mnist"
a = arch.Architecture(architecture, dataset)
steps = 10
batch_size = 256
features = df_features(steps)
merged_df = pd.read_pickle(df_path(steps))
print(merged_df.label.value_counts())
corrected_df = correct_df(merged_df)

print("DF after correcting class imbalance:")
print(corrected_df.label.value_counts())

# Logistic regression
X, Y = f.featurize_db(corrected_df, features)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)
clf = LogisticRegression(random_state=42).fit(X_train, Y_train)
feature_coefs = dict(zip(features, clf.coef_[0]))
y_pred = clf.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print(f"Test Set Accuracy: {accuracy * 100:.2f}%")

pprint(feature_coefs)


def mask_predictor(X, **kwargs):
    return clf.predict(X, **kwargs)


# Debugging why outputs for model are so low
model = a.get_model_constructor()()
masks = [np.ones_like(w) for w in model.get_weights()]
X = meta.make_x(architecture, dataset, model, masks, features,
                train_steps=steps, batch_size=batch_size)
predictions = pd.DataFrame({"predictions": mask_predictor(X)})
print(predictions["predictions"].value_counts())
print(predictions.describe())

merged_df.columns
corrected_df.columns

# X, Y = f.featurize_db(corrected_df, features)
# depth = 1
# width = 32
# meta_model = meta.create_meta(X[0].shape, depth, width)
# history, loss, accuracy = meta.train_meta(meta_model, X, Y, epochs=5)
# pred = meta_model.predict(X, batch_size=2**20)
# print("Mean:", np.mean(pred), "Median:",
#       np.median(pred), "Std:", np.std(pred))
reload(meta)
masks, accuracies = meta.make_meta_mask(
    mask_predictor, meta.make_x, architecture, dataset, 100, 10, features)


# Analyze types of errors made
