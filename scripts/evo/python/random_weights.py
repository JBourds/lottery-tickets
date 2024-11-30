import os
import pickle
import functools
import numpy as np
import numpy.typing as npt
import sys
from typing import Dict, List, Tuple
import tensorflow as tf
from tensorflow import keras
from typing import List, Tuple
sys.path.append(os.path.join(os.path.expanduser("~"), "lottery-tickets"))

import src.harness.evolution as evo

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

hidden_layer_sizes = []
hidden_layer_activations = []

layers = list(zip(hidden_layer_sizes, hidden_layer_activations))

individual_constructor = functools.partial(
    evo.Individual, 
    architecture_name="lenet",
    dataset_name="mnist",
    model_feature_selectors=model_feature_selectors,
    arch_feature_selectors=arch_feature_selectors,
    layers=layers,
)

def random_reinitialize(num_generations: int, steps: int) -> Tuple[npt.NDArray[np.float64]]:
    accuracies = np.zeros(num_generations)
    sparsities = np.zeros(num_generations)
    individual = individual_constructor()
    for generation_index in range(num_generations):
        for _ in range(steps):
            evo.Individual.update_phenotype(individual)
        accuracy = evo.Individual.eval_accuracy(individual)
        sparsity = evo.Individual.sparsity(individual)
        accuracies[generation_index] = accuracy
        sparsities[generation_index] = sparsity
        individual.reinitialize(seed=generation_index)
        individual.clear_metrics()
    return accuracies, sparsities

generations = 1000
steps = 1
random_accuracies, random_sparsities = random_reinitialize(generations, steps)

np.save(f"random_accuracies_{generations}_gens_{steps}_steps", random_accuracies)
np.save(f"random_sparsities_{generations}_gens_{steps}_steps", random_sparsities)
        
    
