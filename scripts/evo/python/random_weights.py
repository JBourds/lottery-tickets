import os
import pickle
import functools
import numpy as np
import numpy.typing as npt
import sys
from typing import Dict, List, Tuple
from tensorflow import keras
from typing import List, Tuple
sys.path.append("~/lottery-tickets")

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

def random_restart(population_size: int, num_generations: int) -> Tuple[npt.NDArray[np.float64]]:
    accuracies = np.zeros((population_size, num_generations))
    sparsities = np.zeros((population_size, num_generations))
    for generation_index in range(num_generations):
        individuals = [individual_constructor() for _ in range(population_size)]
        for individual_index, individual in enumerate(individuals):
            evo.Individual.update_phenotype(individual)
            accuracy = evo.Individual.eval_accuracy(individual)
            sparsity = evo.Individual.sparsity(individual)
            accuracies[individual_index][generation_index] = accuracy
            sparsities[individual_index][generation_index] = sparsity
    return accuracies, sparsities

random_accuracies, random_sparsities = random_restart(50, 1000)
np.save("random_accuracies", random_accuracies)
np.save("random_sparsities", random_sparsities)
        
    
