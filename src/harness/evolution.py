"""
evolution.py

Module containing harness code for running evolutionary computation
experiments with lottery tickets.

Author: Jordan Bourdeau
"""

from src.harness import architecture as arch
from src.harness import utils

import copy
from enum import Enum
import functools
import itertools
from matplotlib import pyplot as plt
import multiprocess as mp
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Any, Callable, Dict, Iterable, List, Literal, Set, Tuple

# Typedefs
Mutation = Callable[[Literal['Individual']], None]
Crossover = Callable[[Literal['Individual'], Literal['Individual']], Literal['Individual']]
FitnessFunction = Callable[[Literal['Individual']], float]
# Feature selectors recursively return a tuple with a boolean flag
# for if the features need to be unpacked, along with a list matching
# the number of features with each element in the list matching the
# dimensionality of the corresponding model/architecture layer
ModelFeatureSelector = Callable[[keras.Model], Tuple[bool, List[np.ndarray]]]
ArchFeatureSelector = Callable[[str, str], Tuple[bool, List[np.ndarray]]]


class ModelFeatures:
    @staticmethod
    def layer_sparsity(model: keras.Model) -> Tuple[bool, List[np.ndarray]]:
        sparsities = [
            nonzero / total 
            for total, nonzero 
            in utils.count_total_and_nonzero_params_per_layer(model)
        ]
        return False, [np.ones_like(w).flatten() * s for s, w in zip(sparsities, model.get_weights())]

    @staticmethod
    def magnitude(model: keras.Model) -> Tuple[bool, List[np.ndarray]]:
        return False, [np.abs(w).flatten() for w in model.get_weights()]

    # Needed to break symmetry so an entire layer is not masked from the beginning
    @staticmethod
    def random(model: keras.Model) -> Tuple[bool, List[np.ndarray]]:
        return False, [np.random.normal(size=w.shape).flatten() for w in model.get_weights()]
    
class ArchFeatures:

    @staticmethod
    def layer_num(architecture_name: str, dataset_name: str) -> Tuple[bool, List[np.ndarray]]:
        a = arch.Architecture(architecture_name, dataset_name)
        weights = a.get_model_constructor()().get_weights()
        return False, [np.ones_like(w).flatten() * i for i, w in enumerate(weights)]
    
    @staticmethod
    def layer_ohe(architecture_name: str, dataset_name: str) -> Tuple[bool, List[List[np.ndarray]]]:
        ohe_layers = arch.Architecture.ohe_layer_types(architecture_name)
        a = arch.Architecture(architecture_name, dataset_name)
        weights = a.get_model_constructor()().get_weights()
        ohe_layer_features = []
        for layer_index, layer_ohe in enumerate(ohe_layers):
            ohe_features = [
                ohe * np.ones_like(weights[layer_index])
                for ohe in layer_ohe
            ]
            if not ohe_layer_features:
                ohe_layer_features = [[layer_ohe_feature] for layer_ohe_feature in ohe_features]
            else:
                for index, layer_ohe_feature in enumerate(ohe_features):
                    ohe_layer_features[index].append(layer_ohe_feature)
        return True, [(False, values) for values in ohe_layer_features]
    
    @staticmethod
    def layer_prop_params(architecture_name: str, dataset_name: str) -> Tuple[bool, List[np.ndarray]]:
        a = arch.Architecture(architecture_name, dataset_name)
        model = a.get_model_constructor()()
        total, _ = utils.count_total_and_nonzero_params(model)
        return False, [w.size / total for w in model.get_weights()]
        

# Individual class representing a NN which maps features about a model and synapses to binary decision 
# for if it will be masked or not
class Individual:
    # One shared copy throughout the class (all individuals in population share same original weights)
    ARCHITECTURE = None
    MODEL = None
    DATA = None
    
    def __init__(
        self, 
        architecture_name: str, 
        dataset_name: str,
        model_features: List[ModelFeatureSelector],
        arch_features: List[ArchFeatureSelector],
        layers: Iterable[Tuple[int, str]],
    ):
        # If this is the first instance of the class, initialize it with read only copies of data
        if self.ARCHITECTURE is None:
            self.ARCHITECTURE_NAME = architecture_name
            self.DATASET_NAME = dataset_name
            self.ARCHITECTURE = arch.Architecture(architecture_name, dataset_name)
            self.MODEL = self.ARCHITECTURE.get_model_constructor()()
            self.MODEL.compile(
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[tf.keras.metrics.CategoricalAccuracy()],
            )
            self.DATA = self.ARCHITECTURE.load_data()
        
        self.model_features = model_features
        self.arch_features = arch_features
        self.arch_feature_values = []
        # Eval this to get the number of outputs which will go into the NN
        # save architecture feature values
        temp_features = self._eval_features()
        from pprint import pprint
        print(type(temp_features[0]))
        pprint(list(map(lambda x: x[0].flatten()[0], temp_features)))
                
        layers = [keras.layers.Input(shape=(len(temp_features),))] \
            + [keras.layers.Dense(size, activation) for size, activation in layers] \
            + [keras.layers.Dense(1, activation='sigmoid')]
        self.genome = keras.Sequential(layers)
        # Dummy loss- we don't train this with gradient descent
        # but use it to map synapse features to probabilities of being masked
        self.genome.compile(loss=tf.keras.losses.CategoricalCrossentropy())
        self._phenotype = None
        self._fitness = None
        self.metrics = {}
        self.rng = np.random.default_rng()
                                                       
    # Private method for evaluating model and architecture features
    def _eval_features(self) -> List[np.ndarray]:
        
        # Helper method which can recursively unpack features
        def unpack_features(unpack_flag: bool, features: List[Tuple[bool, Any]]) -> List[np.ndarray]:
            values = []
            if unpack_flag:
                unpacked_values = [unpack_features(flag, feature_values) for flag, feature_values in features]
                for new_values in unpacked_values:
                    values.extend(new_values)
            else:
                values.append([feature_values for feature_values in features])
            return values
        
        model_features = []
        for feature in self.model_features:
            unpack_flag, feature_values = feature(self.MODEL)
            model_features.extend(unpack_features(unpack_flag, feature_values))
        # Only evaluate this once in an object lifetime
        if not self.arch_feature_values:
            arch_features = []
            for feature in self.arch_features:
                unpack_flag, feature_values = feature(self.ARCHITECTURE_NAME, self.DATASET_NAME)
                arch_features.extend(unpack_features(unpack_flag, feature_values))
            self.arch_feature_values = arch_features
        return model_features + self.arch_feature_values
        
    @staticmethod
    def copy_from(individual: Literal['Individual']) -> Literal['Individual']:
        copied = copy.deepcopy(individual)
        copied.metrics.clear()
        copied._phenotype = None
        copied._fitness = None
        copied.rng = np.random.default_rng()
        return copied
    
    @property
    def phenotype(self) -> List[np.ndarray[bool]]:
        """
        Function which produces a list of boolean Numpy arrays matching the dimensionality
        of the architecture it is trained on based on the output of the NN genotype encoding
        from the computed features for each synapse.
        """
        if self._phenotype is None:
            computed_features = [compute_feature(self.model) for compute_feature in self.features]
            masks = []
            for layer, shape in zip(zip(*computed_features), map(np.shape, self.model.get_weights())):
                X = np.array(list(zip(*layer)))
                mask = (self.genome(X).numpy().reshape(shape) > .5).astype(np.int8)
                masks.append(mask)
            self._phenotype = masks
        return self._phenotype

    @property
    def architecture(self) -> arch.Architecture | None:
        return self.ARCHITECTURE
    
    @property
    def model(self) -> keras.Model | None:
        return self.MODEL
    
    @property
    def data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        return self.DATA
    
    @property
    def training_data(self) -> Tuple[np.ndarray, np.ndarray] | None:
        if self.data is not None:
            X_train, _, Y_train, _ = self.data
            return X_train, Y_train
        
    @property
    def test_data(self) -> Tuple[np.ndarray, np.ndarray] | None:
        if self.data is not None:
            _, X_test, _, Y_test = self.data
            # For now, use a smaller portion for proof of concept
            return X_test[:100], Y_test[:100]
        
    @property
    def fitness(self) -> Any:
        return self._fitness
        
    def copy_model(self) -> keras.Model | None:
        if self.model is not None:
            return copy.deepcopy(self.model)
        
    @staticmethod
    def sparsity(individual: Literal['Individual']) -> float:
        total, nonzero = utils.count_total_and_nonzero_params_from_weights(individual.phenotype)
        return nonzero / total
        
    @staticmethod
    def eval_accuracy(individual: Literal['Individual'], verbose: int = 0) -> float:
        if individual._fitness is None:
            model = individual.copy_model()
            weights = [w * m for w, m in zip(individual.model.get_weights(), individual.phenotype)]
            model.set_weights(weights)
            X_test, Y_test = individual.test_data
            loss, accuracy = model.evaluate(X_test, Y_test, batch_size=len(X_test), verbose=verbose)
            individual._fitness = accuracy
        
        return individual._fitness
    
    # Mutation Methods
    
    @staticmethod
    def get_annealing_mutate():
        def f(
            individual: Literal['Individual'], 
            rate: Callable[[int], float], 
            scale: Callable[[int], float]
        ):
            f.n += 1
            Individual.mutate(individual, rate(f.n), scale(f.n))
        f.n = 0
        return f
    
    @staticmethod
    def mutate(individual: Literal['Individual'], rate: float, scale: float):
        weights = individual.genome.get_weights()
        for layer_index, layer in enumerate(weights):
            perturb_mask = (np.random.uniform(
                low=0, 
                high=1, 
                size=layer.shape,
            ) < rate).astype(np.int8)
            perturbations = -np.abs(individual.rng.normal(
                loc=0,
                scale=scale,
                size=layer.shape,
            )) * perturb_mask
            weights[layer_index] = layer + perturbations
        individual.genome.set_weights(weights)
        individual._phenotype = None
    
    # Crossover Methods
    
    @staticmethod
    def crossover(p1: Literal['Individual'], p2: Literal['Individual']) -> Iterable[Literal['Individual']]:
        child1, child2 = list(map(Individual.copy_from, (p1, p2)))
        p1_weights = p1.genome.get_weights()
        p2_weights = p2.genome.get_weights()
        c1_weights = child1.genome.get_weights()
        c2_weights = child2.genome.get_weights()
        for layer_index, weights in enumerate(p1_weights):
            # Generate a 0/1 for each row, then extend it across all outgoing synapses
            parents = np.repeat(
                np.random.randint(low=0, high=2, size=weights.shape[0]),
                1 if weights.ndim == 1 else weights.shape[1],
                axis=0,
            ).reshape((weights.shape))
            inverse_parents = np.logical_not(parents).astype(np.int8)
            
            # This multiplication uses masks to perform selection
            c1_weights[layer_index] = p1_weights[layer_index] * parents \
                + p2_weights[layer_index] * inverse_parents
            c2_weights[layer_index] = p2_weights[layer_index] * parents \
                + p1_weights[layer_index] * inverse_parents
        child1.genome.set_weights(c1_weights)
        child2.genome.set_weights(c2_weights)
        return child1, child2

# Multi-objective Pareto optimization functions
    
GenomeMetricCallback = Callable[[Dict, List[Individual]], Any]
ObjectiveRangeFunction = Callable[[List[Individual]], float]

class Target(Enum):
    MINIMIZE = 0
    MAXIMIZE = 1
    
ObjectiveFunc = Tuple[Target, ObjectiveRangeFunction, FitnessFunction]
Objective = Tuple[Target, float, FitnessFunction]
    
def pareto_dominates(
    a: Individual,
    b: Individual,
    objectives: List[Objective],
) -> List[bool]:
    for objective, _, fitness in objectives:
        a_fitness, b_fitness = map(
            lambda x: fitness(x) if objective == Target.MAXIMIZE else -fitness(x), 
            (a, b)
        )
        if a_fitness < b_fitness:
            return False
    return True

def ranked_pareto_fronts(
    population: List[Individual],
    objectives: List[Objective],
) -> List[List[Individual]]:
    pop = set(population)
    fronts = []
    while len(pop) > 0:
        next_front = pareto_front(pop, objectives)
        fronts.append(next_front)
        pop -= set(next_front)
    return fronts
    

def pareto_front(
    population: List[Individual],
    objectives: List[Objective],
) -> List[Individual]:
    front = set()
    for individual in population:
        front.add(individual)
        for opponent in front - {individual}:
            if pareto_dominates(opponent, individual, objectives):
                front.remove(individual)
                break
            elif pareto_dominates(individual, opponent, objectives):
                front.remove(opponent)
    return list(front)

# Returns front with sparsities assigned
# Objectives has objective, range, and fitness function
def pareto_front_sparsity(
    front: Iterable[Individual],
    objectives: List[Objective],
) -> Tuple[List[List[Individual]], List[List[float]]]:
    individuals = list(front)
    indices = list(range(len(individuals)))
    sparsities = [0] * len(front)
    
    for obj, obj_range, fitness in objectives:
        reverse = False if obj == Target.MAXIMIZE else False
        key = lambda pop_index: fitness(individuals[pop_index])
        sorted_indices = list(sorted(indices, key=key, reverse=reverse))
        
        sparsities[sorted_indices[0]] = np.inf
        sparsities[sorted_indices[-1]] = np.inf
        for sorted_index, pop_index in enumerate(sorted_indices[1:-1], start=1):
            before = fitness(individuals[sorted_indices[sorted_index - 1]])
            after = fitness(individuals[sorted_indices[sorted_index + 1]])
            sparsities[pop_index] += (after - before) / obj_range
            
    return individuals, sparsities  

# Modified to be able to return multiple winners in each iteration
def nondominated_lexicographic_tournament_selection(
    ranked_fronts: List[List[Individual]],
    sparsities: List[List[float]],
    tournament_size: int,
    num_winners: int,
) -> List[Individual]:
    print(f"Selection with {len(ranked_fronts)} fronts and {len(sparsities)} sparsities") 
    # Returns (front, Individual, sparsity)
    def sample_individual() -> Tuple[int, Individual, float]:
        front_index = np.random.randint(0, len(ranked_fronts))
        individual_index = np.random.randint(0, len(ranked_fronts[front_index]))
        return front_index, ranked_fronts[front_index][individual_index], sparsities[front_index][individual_index]
    
    # Minimize front and maximize sparsity/uniqueness- sorting will put these at start of list
    def key(tup: Tuple[int, Individual, float]) -> Tuple[int, float]:
        front, individual, sparsity = tup
        return front, -sparsity
    
    n_best = sorted(
        [sample_individual() for _ in range(tournament_size)], 
        key=key,
    )
    
    return [individual for _, individual, _ in n_best[:num_winners]]
    
def nsga2(
    num_generations: int,
    archive_size: int,
    population_size: int,
    fronts_to_consider: int,
    tournament_size: int,
    num_tournament_winners: int,
    individual_constructor: Callable[[], Individual],
    objectives: List[ObjectiveFunc],
    crossover: Crossover | None = None,
    mutations: List[Mutation] = [],
    genome_metric_callbacks: List[GenomeMetricCallback] = [],
) -> Tuple[Dict, Dict, List[Individual]]:
    if num_tournament_winners > tournament_size:
        raise ValueError("Cannot have more tournament winners than participants")
        
    # Save data about whole genome and specific objectives over time
    genome_metrics = {}
    objective_metrics = {
        f"objective_{i}_value": np.zeros((population_size, num_generations))
        for i in range(len(objectives))
    }
    objective_metrics.update({
        f"objective_{i}_range": np.zeros(num_generations)
        for i in range(len(objectives))
    })
    
    # Create and evaluate the initial population
    population = [individual_constructor() for _ in range(population_size)]
    archive = []
    
    for generation_index in range(num_generations):
        print(f"Generation {generation_index + 1}")
        
        # Elitist (µ + λ) style strategy
        for individual in population:
            for _, _, fitness in objectives:
                fitness(individual)
        population.extend(archive)
        
        # Create ranked Pareto fronts with their sparsities up to the 
        # number of fronts specified
        concrete_objectives = [(o, r(population), f) for o, r, f in objectives]
        ranked_fronts_with_sparsities = [
            pareto_front_sparsity(front, concrete_objectives)
            for front in ranked_pareto_fronts(population, objectives)[:fronts_to_consider]
        ]
        # Rebuild the archive from the best, most sparse individuals in lower fronts
        archive = []
        ranked_fronts = [tup[0] for tup in ranked_fronts_with_sparsities]
        ranked_sparsities = [tup[1] for tup in ranked_fronts_with_sparsities]
        for front, sparsities in zip(ranked_fronts, ranked_sparsities):
            remaining_spots = archive_size - len(archive)
            if remaining_spots < len(front):
                indices = sorted(
                    list(range(len(sparsities))), 
                    key=lambda i: sparsities[i], 
                    reverse=True,
                )[:remaining_spots]
                archive.extend([front[i] for i in indices])
                break
            else:
                archive.extend(front)
        
        children = []
        # Selection, breeding, and mutation
        selected = nondominated_lexicographic_tournament_selection(
            ranked_fronts, 
            ranked_sparsities,
            tournament_size,
            num_tournament_winners,
        )
        while len(children) < population_size:
            parents = np.random.choice(selected, 2)
            new_children = crossover(*parents) if crossover else list(map(Individual.copy_from, parents))
            
            for child in new_children:
                for mutation in mutations:
                    mutation(child)
            children.extend(new_children)
        population = children[:population_size]
        
        # Callbacks to gather data during training process
        for callback in genome_metric_callbacks:
            callback(genome_metrics, population)
        for obj_index, (_, range_func, fitness_func) in enumerate(objectives):
            objective_metrics[f"objective_{obj_index}_range"][generation_index] = range_func(population)
            for pop_index, individual in enumerate(population):
                objective_metrics[f"objective_{obj_index}_value"][pop_index, generation_index] = fitness_func(individual)
        
    return genome_metrics, objective_metrics, archive

# Metrics over a population

class Population:

    @staticmethod
    def average_sparsity(data: Dict, population: List[Individual]):
        overall_key = "average_global_sparsity"
        layer_key = "average_layer_sparsity"
        for key in [overall_key, layer_key]:
            if data.get(key) is None:
                data[key] = []

        global_counts = list(map(lambda i: utils.count_total_and_nonzero_params_from_weights(i.phenotype), population))
        overall_sparsities = [nonzero / total for total, nonzero in global_counts]
        layer_counts = list(map(lambda i: utils.count_total_and_nonzero_params_per_layer_from_weights(i.phenotype), population))
        layer_sparsities = [[nonzero / total for total, nonzero in layer] for layer in layer_counts]

        data[overall_key].append(np.mean(overall_sparsities))
        data[layer_key].append(np.mean(layer_sparsities, axis=0))
