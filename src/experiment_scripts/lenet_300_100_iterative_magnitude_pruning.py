"""
lenet_300_100_iterative_magnitude_pruning.py


Author: Jordan Bourdeau
Date Created: 4/30/24
"""

import argparse
import functools
import numpy as np
import os
from tensorflow import keras
import sys

sys.path.append('../../')
from src.harness import constants as C
from src.harness import dataset as ds
from src.harness import experiment
from src.harness import model as mod
from src.harness import pruning
from src.harness import rewind

def get_lenet_300_100_experiment_parameters(
    dataset: ds.Datasets,
    first_step_pruning: float, 
    target_sparsity: float, 
    verbose: bool,
    ) -> callable:
    """
    Function which produces a function that conforms to taking in a seed and 
    directory then produces all the output parameters for an experiment.

    Args:
        dataset (ds.Datasets, optional): Dataset being used. Defaults to ds.Datasets.MNIST.
        first_step_pruning (float, optional): % to use for first step of pruning.
        target_sparsity (float, optional): Target sparsity to get below.
        directory (str): Path the experiment is being performed in.

    Returns:
        callable: Function which takes in a seed and directory for a specific run of an experiment
            and produces a dictionary of all parameters to pass into an experiment.
    """
    
    def inner_function(seed: int, directory: str) -> dict:        
        target_dataset: ds.Dataset = ds.Dataset(dataset)
        make_model: callable = functools.partial(
            mod.create_lenet_300_100, 
            target_dataset.input_shape,
            target_dataset.num_classes
        )
        
        # Pruning Parameters- Set parameters, make a reference model, and extract sparsities
        model: keras.Model = make_model()
        sparsities: list[float] = pruning.get_sparsity_percents(model, first_step_pruning, target_sparsity)
            
        pruning_rule: callable = pruning.low_magnitude_pruning
        rewind_rule: callable = rewind.get_rewind_to_original_init_for(seed, directory)
        
        experiment_arguments: dict[str: any] = {
            'random_seed': seed,
            'create_model': make_model,
            'dataset': target_dataset,
            'sparsities': sparsities,
            'pruning_rule': pruning_rule,
            'rewind_rule': rewind_rule,
            'loss_function': None,
            'optimizer': None,
            'global_pruning': False,
            'experiment_directory': directory,
            'verbose': verbose,
        }
        
        return experiment_arguments
    
    return inner_function

if __name__ == '__main__':
    # Default parameters
    experiment_directory: str = os.path.join('../..', C.EXPERIMENTS_DIRECTORY,  'lenet_300_100_iterative_magnitude_pruning_experiment')
    starting_seed: int = 0
    num_experiments: int = 1
    num_batches: int = 1
    first_step_pruning_percent: float = 0.2
    target_sparsity: float = 0.85
    dataset: str = 'mnist'
    max_processes: int = os.cpu_count()
    verbose: bool = False
    
    # Initialize ArgumentParser
    parser = argparse.ArgumentParser(description='Description of your program')

    # Add arguments
    parser.add_argument('--dir', type=str, default=experiment_directory, help='Output directory to store all models and experiment summary.')
    parser.add_argument('--seed', type=int, default=starting_seed, help='Starting seed. Defaults to 0.')
    parser.add_argument('--num_experiments', type=int, default=num_experiments, help='Number of experiments. Defaults to 1.')
    parser.add_argument('--num_batches', type=int, default=num_batches, help='Number of batches to split training into')
    parser.add_argument('--pruning_percent', type=float, default=first_step_pruning_percent, help='First step pruning percent. Defaults to 20%.')
    parser.add_argument('--target_sparsity', type=float, default=target_sparsity, help='Target sparsity. Defaults to 85% sparse (1 step).')
    parser.add_argument('--dataset', type=str, default=dataset, help='Dataset to use for training. Defaults to MNIST.')
    parser.add_argument('--max_processes', type=int, default=max_processes, help='Max number of processes to run in tandem. Defaults to total number of CPU cores.')
    parser.add_argument('--verbose', type=bool, default=verbose, help='Display console output or not.')
    
    # Parse arguments
    args = parser.parse_args()

    # Assign parsed arguments to variables
    experiment_directory = args.dir
    starting_seed = args.seed
    num_experiments = args.num_experiments
    num_batches = args.num_batches
    first_step_pruning_percent = args.pruning_percent
    target_sparsity = args.target_sparsity
    dataset = args.dataset
    verbose = args.verbose
    max_processes = args.max_processes
    
    get_experiment_parameters: callable = get_lenet_300_100_experiment_parameters(
        dataset, 
        first_step_pruning_percent, 
        target_sparsity, 
        verbose,
    )
    
    # Perform paralellized training in evenly split batches
    num_experiments_in_batch: int = int(np.ceil(num_experiments / num_batches))
    for batch_idx in range(num_batches):
        batch_starting_seed: int = starting_seed + batch_idx * num_experiments_in_batch
        batch_directory: str = os.path.join(experiment_directory, f'batch_{batch_idx}')
        
        # Last batch could be smaller
        if batch_idx == num_batches - 1:
            num_experiments_in_batch = num_experiments - batch_idx * num_experiments_in_batch
        
        experiment.run_experiments(
            starting_seed=batch_starting_seed,
            num_experiments=num_experiments_in_batch, 
            experiment_directory=batch_directory,
            experiment=experiment.run_iterative_pruning_experiment,
            get_experiment_parameters=get_experiment_parameters,
            max_processes=max_processes,
        )
    