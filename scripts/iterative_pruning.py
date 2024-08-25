"""
scripts/iterative_pruning.py

Command line script used to run experiments.

Author: Jordan Bourdeau
"""

import argparse

from scripts.base import run_parallel_experiments
from src.harness import pruning
from src.harness.architecture import Architecture

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--dir', type=str, default='lenet-300-100',
                        help='Output directory to store all models and experiment summary.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Starting seed. Defaults to 0.')
    parser.add_argument('--num_experiments', type=int, default=1,
                        help='Number of experiments. Defaults to 1.')
    parser.add_argument('--num_batches', type=int, default=1,
                        help='Number of batches to split training into')
    parser.add_argument('--pruning_percent', type=float, default=0.2,
                        help='First step pruning percent. Defaults to 20%.')
    parser.add_argument('--target_sparsity', type=float, default=0.85,
                        help='Target sparsity. Defaults to 85% sparse (1 step).')
    parser.add_argument('--model', type=str, default='lenet',
                        help='String for the model architecture being used. Defaults to "lenet".')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='Dataset to use for training. Defaults to "mnist". "cifar" is another option.')
    parser.add_argument('--rewind_rule', type=str, default='oi',
                        help='Rule for rewinding weights. "oi" rewinds to original weight initialization.')
    parser.add_argument('--pruning_rule', type=str, default='lm',
                        help='Rule for pruning weights. "lm" prunes low magnitude weights.')
    parser.add_argument('--max_processes', type=int, default=None,
                        help='Max number of processes to run in tandem. Defaults to total number of CPU cores.')
    parser.add_argument('--log_level', type=int, default=2,
                        help='Logging level to use. 0 = Not Set, 1 = Debug, 2 = Info, 3 = Warning, 4 = Error, 5 = Critical.')
    parser.add_argument('--global_pruning', type=bool, default=False,
                        help='Boolean flag for whether to use global pruning. False by default.')

    args = parser.parse_args()
    architecture = Architecture(args.model, args.dataset)
    model = architecture.get_model_constructor()()
    hyperparameters = architecture.get_model_hyperparameters()
    sparsity_percents = pruning.get_sparsity_percents(
        model, args.pruning_percent, args.target_sparsity)

    run_parallel_experiments(
        experiment_directory=args.dir,
        starting_seed=args.seed,
        num_experiments=args.num_experiments,
        num_batches=args.num_batches,
        sparsity_percents=sparsity_percents,
        model=args.model,
        hyperparameters=hyperparameters,
        dataset=args.dataset,
        rewind_rule=args.rewind_rule,
        pruning_rule=args.pruning_rule,
        max_processes=args.max_processes,
        log_level=args.log_level,
        global_pruning=args.global_pruning,
    )
