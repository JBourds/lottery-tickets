"""
scripts/iterative_pruning.py

Command line script used to run experiments.

Author: Jordan Bourdeau
"""

import argparse

from scripts.training.python.base import run_experiments
from scripts.plotting.python.base import make_plots
from src.harness import pruning
from src.harness.architecture import Architecture

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Iterative pruning training script.')
    # Hyperparams
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs per training loop.')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size to use when training.')
    parser.add_argument('--patience', type=int, default=None,
                        help='Number of validations to run without seeing improvement before early stopping (if allowed).')
    parser.add_argument('--minimum_delta', type=float, default=None,
                        help='Minimum amount of loss improvement between two epochs to actually be considered an improvement.')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate.')
    # TODO: Add these as command line arguments
    # parser.add_argument('--optimizer', type=str, default=None,
    #                     help='Optimizer algorithm to use.')
    # parser.add_argument('--accuracy', type=str, default=None,
    #                     help='Accuracy metric to use.')
    # parser.add_argument('--loss', type=str, default=None,
    #                     help='Loss function to use.')
    parser.add_argument('--eval_freq', type=int, default=100,
                        help='Frequency to test on validation data.')
    parser.add_argument('--early_stopping', type=bool, default=True,
                        help='Allow training to prematurely exit if improvements are not detected in performance.')

    # Experiment params
    parser.add_argument('--rundir', type=str, default=None,
                        help='Directory to run in.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Starting seed. Defaults to 0.')
    parser.add_argument('--experiments', type=int, default=1,
                        help='Number of experiments. Defaults to 1.')
    parser.add_argument('--target_sparsity', type=float, default=0.85,
                        help='Target sparsity. Defaults to 85% sparse (1 step).')
    parser.add_argument('--sparsity_strategy', type=str, default='default',
                        help='Sparsity strategy for each round of iterative pruning. Defaults to strategy used in the original paper.')
    parser.add_argument('--model', type=str, default='lenet',
                        help='String for the model architecture being used. Defaults to "lenet".')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='Dataset to use for training. Defaults to "mnist". "cifar" is another option.')
    parser.add_argument('--rewind_rule', type=str, default='oi',
                        help='Rule for rewinding weights. "oi" rewinds to original weight initialization.')
    parser.add_argument('--pruning_rule', type=str, default='lm',
                        help='Rule for pruning weights. "lm" prunes low magnitude weights.')
    parser.add_argument('--seeding_rule', type=str, default='',
                        help='Rule for "seeding" weights at initialization. Check seeding module for documentation.')
    parser.add_argument('--global_pruning', type=bool, default=False,
                        help='Boolean flag for whether to use global pruning. False by default.')

    # Logging
    parser.add_argument('--log_level', type=int, default=2,
                        help='Logging level to use. 0 = Not Set, 1 = Debug, 2 = Info, 3 = Warning, 4 = Error, 5 = Critical.')

    args, unknown = parser.parse_known_args()
    architecture = Architecture(args.model, args.dataset)
    model = architecture.get_model_constructor()()
    cmdline_hyperparams = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'patience': args.patience,
        'minimum_delta': args.minimum_delta,
        'learning_rate': args.lr,
        'eval_freq': args.eval_freq,
        'early_stopping': args.early_stopping,
    }
    cmdline_hyperparams = {key: val for key,
                           val in cmdline_hyperparams.items() if val is not None}
    hyperparameters = architecture.get_model_hyperparameters(
        **cmdline_hyperparams)

    run_experiments(
        path=args.rundir,
        starting_seed=args.seed,
        num_experiments=args.experiments,
        target_sparsity=args.target_sparsity,
        sparsity_strategy=args.sparsity_strategy,
        model=args.model,
        hyperparameters=hyperparameters,
        dataset=args.dataset,
        rewind_rule=args.rewind_rule,
        pruning_rule=args.pruning_rule,
        seeding_rule=args.seeding_rule,
        log_level=args.log_level,
        global_pruning=args.global_pruning,
    )

    make_plots(args.rundir, seeding_rule=args.seeding_rule)
        
