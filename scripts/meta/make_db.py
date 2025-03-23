import argparse
import tensorflow as tf
from copy import deepcopy
from copy import copy as shallowcopy
from tensorflow import keras
import numpy.typing as npt
from importlib import reload
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
from pprint import pprint
import os
from tqdm import tqdm
from typing import Dict, Generator, List, Tuple
import sys
root = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))))
sys.path.append(root)

tf.keras.backend.set_floatx('float64')


if __name__ == "__main__":
    from src.metrics.synflow import compute_synflow_per_weight
    from src.metrics import features as f
    from src.harness import history as hist
    from src.harness import meta
    from src.harness import dataset as ds
    from src.harness import architecture as arch
    epath = "/users/j/b/jbourde2/lottery-tickets/11-04-2024/lenet_mnist_0_seed_5_experiments_1_batches_0.025_default_sparsity_lm_pruning_20241102-111614"
    parser = argparse.ArgumentParser(
        prog="Make DB", description="Create a Pandas database with weights and weight features from lottery ticket experiment output.")
    parser.add_argument(
        "-b", "--batch", help="Batch size to use.", default=64, type=int)
    parser.add_argument(
        "-e", "--epath", help="Experiment directory path.", default=epath)
    parser.add_argument(
        "-s", "--steps", help="Number of training steps.", type=int, default=1)

    args = parser.parse_args()
    steps = int(args.steps)

    def make_db(steps: int):
        trials_df, layers_df, weights_df, trained_weights_df = f.build_dataframes(
            args.epath, train_steps=steps, batch_size=args.batch)
        merged_df = f.merge_dfs(trials_df, layers_df,
                                weights_df, trained_weights_df)
        data_dir = os.path.join(args.epath, "data_old")
        os.makedirs(data_dir, exist_ok=True)
        output_path = os.path.join(data_dir, f"merged_{args.steps}_steps.pkl")
        merged_df.to_pickle(output_path)

    make_db(steps)
