"""
experiment.py

Module containing code for actually running the lottery ticket hypothesis experiemnts.

Modified By: Jordan Bourdeau
Date: 3/17/24
"""

import functools
import numpy as np
import tensorflow as tf
from tensorflow import keras

import src.harness.constants as C
import src.harness.model as mod
import src.harness.pruning as pruning
import src.harness.rewind as rewind
import src.harness.training as train
import src.harness.utils as utils

class ExperimentData:
    def __init__(self):
        """
        Class which stores the data from an experiment 
        (list of `TrainingRound` objects which is of length N, where N is the # of pruning steps).
        """
        self.pruning_rounds: list[train.TrainingRound] = []

    def add_pruning_round(self, round: train.TrainingRound):
        """
        Method used to add a `TrainingRound` object to the internal representation.

        :param round: `TrainingRound` object being added.
        """
        self.pruning_rounds.append(round)


class ExperimentSummary:
    def __init__(self):
        """
        Class which stores data from many experiments in a dictionary, where the key
        is the random seed used for the experiment and the value is an `ExperimentData` object.
        """
        self.experiments: dict[int: ExperimentData] = {}

    def add_experiment(self, seed: int, experiment: ExperimentData):
        """
        Method to add a new experiment to the internal dictionary.

        :param seed:        Integer for the random seed used in the experiment.
        :param experiment: `Experiment` object to store.
        """
        self.experiments[seed] = experiment

    def __str__(self) -> str:
      """
      String representation to create a summary of the experiment.

      :returns: String representation.
      """
      for seed, experiment in self.experiments.items():
          print(f'\nSeed {seed}')
          for idx, round in enumerate(experiment.pruning_rounds):
              print(f'Pruning Step {idx}:')
              print(round)

def run_iterative_pruning_experiment(
    random_seed: int, 
    create_model: callable, 
    make_dataset: callable,
    sparsities: list[float], 
    pruning_rule: callable = None,
    rewind_rule: callable = None,
    loss_function: callable = None,
    optimizer: tf.keras.optimizers.Optimizer = None, 
    global_pruning: bool = False,
    num_epochs: int = C.TRAINING_EPOCHS,
    batch_size: int = C.BATCH_SIZE,
    patience: int = C.PATIENCE,
    minimum_delta: float = C.MINIMUM_DELTA,
    allow_early_stopping: bool = True,
    verbose: bool = True,
    ) -> ExperimentData:
    """
    Function used to run the pruning experiements for a given random seed.
    Will perform iterative pruning given a specified pruning technique
    (e.g. Magnitude-based pruning) and a given rewind technique
    (e.g. Rewind to initial weights).

    Args:
        random_seed (int): Random seed to use for reproducable results.
        create_model (callable): Function used to produce/initialize the model.
        sparsities (list[float]): List of perentages (0 to 1) for sparsities 
            at each step of pruning.
        verbose (bool): _description_

    Returns:
        ExperimentData: Class containing all the training round data in a list.
    """
    # Set seet for reproducability
    utils.set_seed(random_seed)
    
    # Handle outer loop default values
    if pruning_rule is None:
        pruning_rule = pruning.low_magnitude_pruning
    if rewind_rule is None:
        rewind_rule = functools.partial(rewind.rewind_to_original_init, random_seed)
    
    experiment_data: ExperimentData = ExperimentData()

    # Make models and save them
    model : keras.Model = create_model()
    mask_model: keras.Model = mod.create_masked_nn(create_model)   
    mod.save_model(model, random_seed, 0, initial=True)
    mod.save_model(mask_model, random_seed, 0, masks=True, initial=True)

    for pruning_step, sparsity in enumerate(sparsities):
        # Prune the model to the new sparsity
        pruning.prune(model, pruning_rule, sparsity, global_pruning=global_pruning)
        
        # Update mask model
        pruning.update_masks(model, mask_model)
        
        # Reset unpruned weights to original values.
        rewind.rewind_model_weights(model, mask_model, rewind_rule)
        
        if loss_function is None:
            loss_function = C.LOSS_FUNCTION()
        if optimizer is None:
            optimizer = C.OPTIMIZER()
        accuracy_metric: tf.keras.metrics.Metric = tf.keras.metrics.CategoricalAccuracy()

        training_round: train.TrainingRound = train.train(
            random_seed, 
            pruning_step, 
            model, 
            mask_model, 
            make_dataset, 
            num_epochs=num_epochs, 
            batch_size=batch_size, 
            patience=patience, 
            minimum_delta=minimum_delta,
            loss_function=loss_function,
            optimizer=optimizer,
            allow_early_stopping=allow_early_stopping,
        )
        experiment_data.add_pruning_round(training_round)

        if verbose:
            print(f'\nTook {np.sum(training_round.test_accuracies != 0)} / {C.TRAINING_EPOCHS} epochs')
            print(f'Ended with a best training accuracy of {np.max(training_round.train_accuracies) * 100:.2f}% and test accuracy of training accuracy of {np.max(training_round.test_accuracies) * 100:.2f}%')
            
    return experiment_data