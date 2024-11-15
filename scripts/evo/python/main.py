
"""
scripts/evo/main.py

Command line script to run evolutionary algorithm experiments
on lottery tickets.

Author: Jordan Bourdeau
"""

import argparse
import functools
import numpy as np
from tensorflow import keras

from src.harness import evolution as evo
from src.harness import history

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evolutionary algorithms script.")
    
    # Feature subset (hardcoded to all features)
    
    # Hidden layers
    parser.add_argument("--hidden_sizes", type=str, default="", help="--hidden_sizes=[n,]")
    parser.add_argument("--hidden_activations", type=str, default="", help="--hidden_activations=[type,]")
    
    # Num experiments
    parser.add_argument("--experiments", type=int, default=1, help="--experiments=n")
    
    # Num generations
    parser.add_argument("--generations", type=int, default=2, help="--generations=n")
    
    # Archive size
    parser.add_argument("--asize", type=int, default=10, help="--asize=n")
    
    # Population size
    parser.add_argument("--psize", type=int, default=10, help="--psize=n")
    
    # Fronts to consider
    parser.add_argument("--nfronts", type=int, default=2, help="--nfronts=n")
    
    # Tournament size
    parser.add_argument("--tsize", type=int, default=10, help="--psize=n")
    
    # Num tournament winners
    parser.add_argument("--nwinners", type=int, default=2, help="--nfronts=n")

    # Architecture and dataset names
    parser.add_argument("--aname", type=str, default="lenet", help="--aname=<arch>")
    parser.add_argument("--dname", type=str, default="mnist", help="--dname=<dataset>")
    
    # Mutation (hardcoded to simulated annealing, allow rate/scale tweaks)
    parser.add_argument("--mscale", type=float, default=0.5, help="--mscale=n")
    parser.add_argument("--mrate", type=float, default=0.2, help="--mrate=n")
    
    # Objectives (hardcoded)
    # Model features (hardcoded)
    # Architecture features (hardcoded)
    
    # Logging
    parser.add_argument("--log_level", type=int, default=2,
                        help="Logging level to use. 0 = Not Set, 1 = Debug, 2 = Info, 3 = Warning, 4 = Error, 5 = Critical.")
    
    args, unknown = parser.parse_known_args()
    hidden_layer_sizes = list(map(int, args.hidden_sizes.split(","))) if args.hidden_sizes else []
    if args.hidden_activations == "":
        hidden_layer_activations = ["relu"] * len(hidden_layer_sizes)
    hidden_layer_activations = args.hidden_activations.split(",") if args.hidden_activations else []
    if len(hidden_layer_sizes) != len(hidden_layer_activations):
        raise ValueError("Hidden layer sizes and activations must have the same number of inputs.")

    model_feature_selectors = [
        evo.ModelFeatures.layer_sparsity, 
        evo.ModelFeatures.magnitude,
        evo.ModelFeatures.random,
        functools.partial(evo.ModelFeatures.synaptic_flow, loss_fn=keras.losses.CategoricalCrossentropy()),
    ]
    arch_feature_selectors = [
        evo.ArchFeatures.layer_num,
        evo.ArchFeatures.layer_ohe,
        evo.ArchFeatures.layer_prop_params,
    ]

    layers = list(zip(hidden_layer_sizes, hidden_layer_activations))

    individual_constructor = functools.partial(
        evo.Individual, 
        architecture_name=args.aname,
        dataset_name=args.dname,
        model_feature_selectors=model_feature_selectors,
        arch_feature_selectors=arch_feature_selectors,
        layers=layers,
    )

    objectives = [
        (evo.Target.MAXIMIZE, lambda x: 1, evo.Individual.eval_accuracy),
        (evo.Target.MINIMIZE, lambda x: 1, evo.Individual.sparsity),
    ]

    rate_func = lambda n: args.mrate
    scale_func = lambda n: args.mscale / np.sqrt(n + 1)
    mutations = [
        functools.partial(evo.Individual.get_annealing_mutate(), rate=rate_func, scale=scale_func),
        evo.Individual.update_phenotype,
    ]

    genome_metric_callbacks = []
    
    kwargs = {
        "num_generations": args.generations,
        "archive_size": args.asize,
        "population_size": args.psize,
        "fronts_to_consider": args.nfronts,
        "tournament_size": args.tsize,
        "num_tournament_winners": args.nwinners,
        "individual_constructor": individual_constructor,
        "objectives": objectives,
        "mutations": mutations,
        "crossover": evo.Individual.crossover,
        "genome_metric_callbacks": genome_metric_callbacks,
    }

    all_genome_metrics = []
    all_objective_metrics = []
    all_archives = []
    for run in range(args.experiments):
        print(f"Run {run + 1}")
        genome_metrics, objective_metrics, archive = evo.nsga2(**kwargs)
        all_genome_metrics.append(genome_metrics)
        all_objective_metrics.append(objective_metrics)
        all_archives.append(archive)
