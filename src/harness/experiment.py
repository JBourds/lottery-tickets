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

from src.harness import constants as C
from src.harness import dataset as ds
from src.harness import model as mod
from src.harness import pruning 
from src.harness import rewind
from src.harness import training as train
from src.harness import utils

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
              
def get_mnist_lenet_300_100_experiment_parameters(
    first_step_pruning: float = 0.20, 
    target_sparsity: float = 0.01, 
    ) -> tuple:
    """
    Function which produces all the parameters for using the LeNet-300-100
    architecture with the MNIST dataset.

    Returns:
        tuple[...]: Positional arguments which will get passed into an experiment function.
    """
    dataset: ds.Dataset = ds.Dataset(ds.Datasets.MNIST)
    make_model: callable = functools.partial(
        mod.create_lenet_300_100, 
        dataset.input_shape,
        dataset.num_classes
    )
    
    # Pruning Parameters- Set parameters, make a reference model, and extract sparsities
    model: keras.Model = make_model()
    sparsities: list[float] = pruning.get_sparsity_percents(model, first_step_pruning, target_sparsity)
    
    return make_model, dataset, sparsities
            
def run_experiments(
    num_experiments: int, 
    get_experiment_parameters: callable, 
    experiment: callable,
    ) -> ExperimentSummary:
    """
    Function where experimental parameters are configured to run.

    Args:
        num_experiments (int): Number of experiments to run with the
            specified configuration.
        get_experiment_parameters (callable): Function which produces all the positional
            and keyword arguments to get passed into the experiment being ran.
        experiment (callable): Function to run the experiments.

    Returns:
        ExperimentSummary: Object containing information about all trained models.
    """
    
    # Object to keep track of experiment data
    experiment_summary: ExperimentSummary = ExperimentSummary()
    
    # For each experiment, use a different random seed and keep track of all the data produced
    for seed in range(num_experiments):
        experiment_data: ExperimentData = experiment(
            seed,
            *get_experiment_parameters()
        )
        experiment_summary.add_experiment(seed, experiment_data)
    return experiment_summary      

def run_iterative_pruning_experiment(
    random_seed: int, 
    create_model: callable, 
    dataset: ds.Dataset,
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
        random_seed (int): Random seed for reproducability.
        create_model (callable): Function which produces the model.
        dataset (enum): Enum for the dataset being used.
        sparsities (list[float]): List of sparsities for each step of training.
        pruning_rule (callable, optional): Function used to prune model. 
            Defaults to None - becomes low magnitude pruning.
        rewind_rule (callable, optional): Function used for rewinding model weights. 
            Defaults to None - becomes initial weights.
        loss_function (callable, optional): Loss function for training. 
            Defaults to None - becomes loss function specified in `constants.py`.
        optimizer (tf.keras.optimizers.Optimizer, optional): Optimizer. 
            Defaults to None - becomes optimizer specified in `constants.py`..
        global_pruning (bool, optional): Boolean flag for whether pruning is done globally
            or layerwise. Defaults to False.
        num_epochs (int, optional): Number of epochs to train for. Defaults to C.TRAINING_EPOCHS.
        batch_size (int, optional): Batch size to use. Defaults to C.BATCH_SIZE.
        patience (int, optional): Number of epochs training will continue for without improvement
            before implementing early stopping. Defaults to C.PATIENCE.
        minimum_delta (float, optional): Minimum positive change required to count as an improvement. 
            Defaults to C.MINIMUM_DELTA.
        allow_early_stopping (bool, optional): Boolean flag for whether early stopping is enabled. 
            Defaults to True.
        verbose (bool, optional): Boolean flag for whether console output is displayed. Defaults to True.

    Returns:
        ExperimentData: Object containing information about all the training rounds produced in the experiment.
    """
    # Set seed for reproducability
    utils.set_seed(random_seed)
    
    # Handle outer loop default values
    if pruning_rule is None:
        pruning_rule = pruning.low_magnitude_pruning
    if rewind_rule is None:
        rewind_rule = functools.partial(rewind.rewind_to_original_init, random_seed)
    
    experiment_data: ExperimentData = ExperimentData()

    # Make models and save them
    model: keras.Model = create_model()
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
            dataset, 
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