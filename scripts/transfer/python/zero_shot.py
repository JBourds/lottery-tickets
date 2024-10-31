"""
zero_shot.py

Test script whch performs zero shot transfer learning by taking
the masks from one trial and putting them onto a model with the
same shape (e.g., MNIST -> Fashion MNIST) to evaluate accuracy.

Author: Jordan Bourdeau
"""

import copy
from matplotlib import pyplot as plt
import numpy as np
import os
from pprint import pprint
import sys
import tensorflow as tf
from typing import Dict

sys.path.append(os.path.join(os.environ["HOME"], "lottery-tickets"))

from src.harness import architecture as arch
from src.harness import dataset as ds
from src.harness import history
from src.harness import utils

def transfer_masking(seed: int, src: str, src_ds: ds.Dataset, dst_ds: ds.Dataset, **kwargs) -> Dict[str, np.ndarray]:
    """
    Function to perform transfer masking and evaluate validation loss/accuracy between the source
    and target datasets using the trained mask. Also performs a masked pass using a set of randomly
    initialized weights as a control.

    @param seed (int): Random seed which was used.
    @param src (str): Source path for an experiment (e.g., "model0").
    @param src_ds (ds.Dataset): Dataset used for the experiment originally.
    @param dst_ds (ds.Dataset): Dataset to evaluate on.
    @param **kwargs: Keyword arguments to pass into get_trials.

    @returns (Dict[str, np.ndarray]): Data dictionary containing experiment results.
    """
    trial_generator = history.get_trials(src, **kwargs) 
    is_first_run = True
    data_dict = {
        "src_lt_loss": [],
        "src_lt_acc": [],
        "dst_lt_loss": [],
        "dst_lt_acc": [],
        "src_rand_loss": [],
        "src_rand_acc": [],
        "dst_rand_loss": [],
        "dst_rand_acc": [],
    }
    
    utils.set_seed(seed)
    for index, trial in enumerate(trial_generator):
        if is_first_run:
            architecture = arch.Architecture(trial.architecture, src_ds)
            _, src_X_test, _, src_Y_test = architecture.load_data()
            architecture = arch.Architecture(trial.architecture, dst_ds)
            _, dst_X_test, _, dst_Y_test = architecture.load_data()
            loader = architecture.get_model_constructor()
            model = loader()
            model.compile(
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[tf.keras.metrics.CategoricalAccuracy()],
            )
            saved_weights = copy.deepcopy(model.get_weights())
            is_first_run = False
        
        model.set_weights([w * m for w, m in zip(saved_weights, trial.masks)])
        # Test with source dataset using trained mask
        loss, accuracy = model.evaluate(src_X_test, src_Y_test)
        data_dict["src_lt_loss"].append(loss)
        data_dict["src_lt_acc"].append(accuracy)
        
        # Test with target dataset using trained mask
        loss, accuracy = model.evaluate(dst_X_test, dst_Y_test)
        data_dict["dst_lt_loss"].append(loss)
        data_dict["dst_lt_acc"].append(accuracy)
        
        utils.set_seed(seed + index + 1)
        random_masks = [np.ones(m.size) for m in trial.masks]
        zeroes_count = [int(m.size - np.sum(m)) for m in trial.masks]
        for m, zeroes_in_layer, source_mask in zip(random_masks, zeroes_count, trial.masks):
            m[:zeroes_in_layer] = 0
            np.random.shuffle(m)
        random_masks = [np.reshape(m, source_mask.shape) for m, source_mask in zip(random_masks, trial.masks)]
        model.set_weights([w * m for w, m in zip(saved_weights, random_masks)])
        # Test with source dataset using random mask
        loss, accuracy = model.evaluate(src_X_test, src_Y_test)
        data_dict["src_rand_loss"].append(loss)
        data_dict["src_rand_acc"].append(accuracy)

        # Test with target dataset using random mask
        loss, accuracy = model.evaluate(dst_X_test, dst_Y_test)
        data_dict["dst_rand_loss"].append(loss)
        data_dict["dst_rand_acc"].append(accuracy)
        
    return data_dict 
        
         
if __name__ == "__main__":
    src = "/users/j/b/jbourde2/lottery-tickets/experiments/lenet_mnist_0_seed_3_experiments_1_batches_0.05_default_sparsity_lm_pruning_20241015-053957/models/model0/"
    plots_dir = "/users/j/b/jbourde2/lottery-tickets/experiments/lenet_mnist_0_seed_3_experiments_1_batches_0.05_default_sparsity_lm_pruning_20241015-053957/plots"
    os.makedirs(plots_dir, exist_ok=True)
    seed = 0
    data = transfer_masking(seed, src, "mnist", "fashion_mnist")
    trial_count = 15
    trials = np.arange(trial_count)
    plt.plot(trials, data["src_lt_acc"], label="Mask on Src")
    plt.plot(trials, data["dst_lt_acc"], label="Mask on Dst")
    plt.plot(trials, data["src_rand_acc"], label="Random Mask on Src")
    plt.plot(trials, data["dst_rand_acc"], label="Random Mask on Dst")
    plt.legend()
    plt.title("Zero Shot Transfer Masking")
    plt.xlabel("Trial Number")
    plt.ylabel("Validation Accuracy")
    plt.gcf().savefig(os.path.join(plots_dir, "accuracies.png"))
