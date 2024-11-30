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
import keras.backend as K
from tensorflow import keras
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Set, Tuple

# Typedefs
Mutation = Callable[[Literal['Individual']], None]
Crossover = Callable[[Literal['Individual'], Literal['Individual']], Literal['Individual']]
FitnessFunction = Callable[[Literal['Individual']], float]

# Feature selectors recursively return a tuple with a boolean flag
# for if the features need to be unpacked, along with a list matching
# the number of features with each element in the list matching the
# dimensionality of the corresponding model/architecture layer
ModelFeatureSelector = Callable[[Literal['Individual']], Tuple[bool, List[np.ndarray]]]
ArchFeatureSelector = Callable[[Literal['Individual']], Tuple[bool, List[np.ndarray]]]

# Multi-objective Pareto optimization functions

class Target(Enum):
    MINIMIZE = 0
    MAXIMIZE = 1
    
GenomeMetricCallback = Callable[[Dict, List[Literal['Individual']]], Any]
ObjectiveRangeFunction = Callable[[List[Literal['Individual']]], float]
ObjectiveFunc = Tuple[Target, ObjectiveRangeFunction, FitnessFunction]
Objective = Tuple[Target, float, FitnessFunction]


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
        model_feature_selectors: List[ModelFeatureSelector],
        arch_feature_selectors: List[ArchFeatureSelector],
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
        
        self._masked_model = None
        # Allocate all this memory up front then mutate it in place from there
        self._phenotype_init = False
        self._phenotype = [np.ones_like(w) for w in self.MODEL.get_weights()]
        self._masked_model = copy.deepcopy(self.MODEL)
        self.metrics = {}
        self.rng = np.random.default_rng()
        self.model_feature_selectors = model_feature_selectors
        self.arch_feature_values = self._eval_features(arch_feature_selectors)
        # This would not work currently if there were any model feature selectors which
        # needed to be unpacked but it's not an issue fo rnow
        num_features = len(self.arch_feature_values) + len(self.model_feature_selectors)
        
        layers = [keras.layers.Input(shape=(num_features,))] \
            + [keras.layers.Dense(size, activation) for size, activation in layers] \
            + [keras.layers.Dense(1, activation='sigmoid')]
        self.genome = keras.Sequential(layers)
        # Dummy loss- we don't train this with gradient descent
        # but use it to map synapse features to probabilities of being masked
        self.genome.compile(loss=tf.keras.losses.CategoricalCrossentropy())
        
        # Allocate a big matrix to store the features for every synapse in
        self.num_synapses = np.sum([w.size for w in self.MODEL.get_weights()])
        self.X = np.zeros((self.num_synapses, num_features))
                
    def _eval_features(self, feature_selectors: List[ModelFeatureSelector | ArchFeatureSelector]) -> List[np.ndarray]:
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
        
        results = []
        for feature in feature_selectors:
            unpack_flag, feature_values = feature(self)
            results.extend(unpack_features(unpack_flag, feature_values))
        return results
        
    @staticmethod
    def copy_from(src: Literal['Individual'], dst: Literal['Individual']):
        dst.metrics.clear()
        dst.rng = np.random.default_rng()
        dst.genome = copy.deepcopy(src.genome)
        # Causes phenotype to be reevaluated next time it is required
        dst._phenotype_init = False
    
    @property
    def phenotype(self) -> List[np.ndarray[np.int8]]:
        """
        Function which produces a list of boolean Numpy arrays matching the dimensionality
        of the architecture it is trained on based on the output of the NN genotype encoding
        from the computed features for each synapse.
        """
        if not self._phenotype_init:
            for m in self._phenotype:
                m = np.ones_like(m)
            self._phenotype_init = True
        return self._phenotype
    
    @property
    def masked_model(self) -> List[np.ndarray]: 
        self._masked_model.set_weights([m * w for m, w in zip(self.phenotype, self.model.get_weights())])
        return self._masked_model
    
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
        
    @staticmethod
    def sparsity(individual: Literal['Individual']) -> float:
        if individual.metrics.get("sparsity") is None:
            total, nonzero = utils.count_total_and_nonzero_params_from_weights(individual.phenotype)
            sparsity = nonzero / total
            individual.metrics["sparsity"] = sparsity
        return individual.metrics["sparsity"]
        
    @staticmethod
    def eval_accuracy(individual: Literal['Individual'], verbose: int = 0) -> float:
        if individual.metrics.get("accuracy") is None:
            X_test, Y_test = individual.test_data
            loss, accuracy = individual.masked_model.evaluate(X_test, Y_test, batch_size=len(X_test), verbose=verbose)
            individual.metrics["accuracy"] = accuracy
        return individual.metrics["accuracy"]
    
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
    
    # Method must be called in order to update the phenotype based on the output of the specified
    # features with the previous phenotype (starting with mask of all 1s)
    @staticmethod
    def update_phenotype(individual: Literal['Individual']):
        computed_features = individual._eval_features(individual.model_feature_selectors) + individual.arch_feature_values
        masks = []
        for layer, shape in zip(zip(*computed_features), map(np.shape, individual.model.get_weights())):
            # Turn this into matrix multiplication for SPEED
            flattened_layers = [l.flatten() for l in layer]
            X = np.array(list(zip(*flattened_layers)))
            mask = (individual.genome(X).numpy().reshape(shape) > .5).astype(np.int8)
            masks.append(mask)
        individual._phenotype = masks
    
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
    
    # Crossover Methods
    
    @staticmethod
    def crossover(
        p1: Literal['Individual'], 
        p2: Literal['Individual'], 
        c1: Literal['Individual'], 
        c2: Literal['Individual'],
    ) -> Iterable[Literal['Individual']]:
        for parent, child in zip((p1, p2), (c1, c2)):
            Individual.copy_from(parent, child)
        p1_weights = p1.genome.get_weights()
        p2_weights = p2.genome.get_weights()
        c1_weights = c1.genome.get_weights()
        c2_weights = c2.genome.get_weights()
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
        c1.genome.set_weights(c1_weights)
        c2.genome.set_weights(c2_weights)
        return c1, c2

    def clear_metrics(self):
        self.metrics.clear()

    def reinitialize(self, seed: int):
        utils.set_seed(seed) 
        # Iterate through the layers of the model
        for layer in self.genome.layers:
            # Check if the layer has weights (e.g., Dense, Conv2D)
            if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
                # Reinitialize weights and biases
                for weight in layer.weights:
                    # Use the layer's original initializer to reinitialize weights
                    initializer = layer.initializer if hasattr(layer, 'initializer') else tf.keras.initializers.GlorotUniform()
                    new_values = initializer(shape=weight.shape, dtype=weight.dtype)
                    weight.assign(new_values)
                

class ModelFeatures:
    @staticmethod
    def layer_sparsity(individual: Individual) -> Tuple[bool, List[np.ndarray]]:
        sparsities = [
            nonzero / total 
            for total, nonzero 
            in utils.count_total_and_nonzero_params_per_layer_from_weights(individual.phenotype)
        ]
        return False, [np.ones_like(w) * s for s, w in zip(sparsities, individual.phenotype)]

    @staticmethod
    def magnitude(individual: Individual) -> Tuple[bool, List[np.ndarray]]:
        return False, [np.abs(w) for w in individual.model.get_weights()]

    # Needed to break symmetry so an entire layer is not masked from the beginning
    @staticmethod
    def random(individual: Individual) -> Tuple[bool, List[np.ndarray]]:
        return False, [np.random.normal(size=w.shape) for w in individual.model.get_weights()]
    
    # Implementation derived from: https://github.com/ganguli-lab/Synaptic-Flow/blob/master/Pruners/pruners.py
    @staticmethod
    def synaptic_flow(
        individual: Individual,
        loss_fn: Optional[keras.losses.Loss],
    ) -> Tuple[bool, List[np.ndarray]]:
        X_test, Y_test = individual.test_data
        X_test, Y_test = tf.convert_to_tensor(X_test), tf.convert_to_tensor(Y_test)
        synaptic_flows = []
        
        model = individual.masked_model
        with tf.GradientTape() as tape:
            predictions = model(X_test, training=True)
            loss = loss_fn(Y_test, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        for grad, param in zip(gradients, model.trainable_variables):
            if grad is not None:
                synaptic_flow = tf.abs(grad * param)
                synaptic_flows.append(synaptic_flow.numpy())
        
        return False, synaptic_flows
    
class ArchFeatures:

    @staticmethod
    def layer_num(individual: Individual) -> Tuple[bool, List[np.ndarray]]:
        architecture_name, dataset_name = individual.ARCHITECTURE_NAME, individual.DATASET_NAME
        a = arch.Architecture(architecture_name, dataset_name)
        weights = a.get_model_constructor()().get_weights()
        return False, [np.ones_like(w) * i for i, w in enumerate(weights)]
    
    @staticmethod
    def layer_ohe(individual: Individual) -> Tuple[bool, List[List[np.ndarray]]]:
        architecture_name, dataset_name = individual.ARCHITECTURE_NAME, individual.DATASET_NAME
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
    def layer_prop_params(individual: Individual) -> Tuple[bool, List[np.ndarray]]:
        architecture_name, dataset_name = individual.ARCHITECTURE_NAME, individual.DATASET_NAME
        a = arch.Architecture(architecture_name, dataset_name)
        model = a.get_model_constructor()()
        total, _ = utils.count_total_and_nonzero_params(model)
        return False, [np.ones_like(w) * (w.size / total) for w in model.get_weights()]
        
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
    
# NOTE: This kept running out of memory and failing so I updated it to not make any new allocations
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
    
    # Pre-allocate the maximum number of individuals which are needed at once,
    # and use this list as a form of allocator which they get pushed/popped from
    # + 1 in case we want an odd population size
    max_individuals_at_a_time = 2 * population_size + 2 * archive_size + 1
    allocations = [individual_constructor() for _ in range(max_individuals_at_a_time)]
    population = [allocations.pop() for _ in range(population_size)]
    archive = []

    best_genome = None
    best_accuracy = None

    def reevaluate_best() -> Individual:
        nonlocal best_accuracy
        nonlocal best_genome

        best = max(population, key=Individual.eval_accuracy)
        # print(f"Current Best Accuracy {best_accuracy}, Genome: {best_genome}")
        if best_genome is None or best_accuracy is None or Individual.eval_accuracy(best) > best_accuracy:
            best_genome = copy.deepcopy(best.genome.get_weights())
            best_accuracy = Individual.eval_accuracy(best)
            # print(f"New Best Accuracy {best_accuracy}, Genome: {best_genome}")
        # print(f"Current Best Accuracy {best_accuracy}, Genome: {best_genome}")

    reevaluate_best()
        
    for generation_index in range(num_generations):
        print(f"Generation {generation_index + 1}")
        # Elitist (µ + λ) style strategy
        # Laterally shift already allocated individuals from archive to population
        population.extend(archive)
        archive = []
        
        # Create ranked Pareto fronts with their sparsities up to the 
        # number of fronts specified
        concrete_objectives = [(o, r(population), f) for o, r, f in objectives]
        ranked_fronts_with_sparsities = [
            pareto_front_sparsity(front, concrete_objectives)
            for front in ranked_pareto_fronts(population, objectives)[:fronts_to_consider]
        ]
        # Rebuild the archive from the best, most sparse individuals in lower fronts
        # Allocates `archive_size` individuals
        ranked_fronts = [tup[0] for tup in ranked_fronts_with_sparsities]
        ranked_sparsities = [tup[1] for tup in ranked_fronts_with_sparsities]
        new_archive = [allocations.pop() for _ in range(archive_size)]
        index = 0
        for front, sparsities in zip(ranked_fronts, ranked_sparsities):
            remaining_spots = archive_size - index
            if remaining_spots < len(front):
                indices = sorted(
                    list(range(len(sparsities))), 
                    key=lambda i: sparsities[i], 
                    reverse=True,
                )[:remaining_spots]
                for src_index, dst in zip(indices, new_archive[index:]):
                    Individual.copy_from(front[src_index], dst)
                    index += 1
                break
            else:
                for src, dst in zip(front, new_archive[index:]):
                    Individual.copy_from(src, dst)
                    index += 1
        archive = new_archive
        
        children = []
        # Selection, breeding, and mutation
        # Selection does not make any new allocations, and will be a view on the population
        selected = nondominated_lexicographic_tournament_selection(
            ranked_fronts, 
            ranked_sparsities,
            tournament_size,
            num_tournament_winners,
        )
        # Each iteration = 2 allocations until we are at `population_size` allocations
        while len(children) < population_size:
            new_children = [allocations.pop() for _ in range(2)]
            parents = np.random.choice(selected, 2)
            crossover(*parents, *new_children)
            
            for child in new_children:
                for mutation in mutations:
                    mutation(child)
            children.extend(new_children)
        # Free the combined old population and archive, as well as any remainders
        allocations.extend(population + children[population_size:])
        population = children[:population_size]

        reevaluate_best()
        
        # Callbacks to gather data during training process
        for callback in genome_metric_callbacks:
            callback(genome_metrics, population)
        for obj_index, (_, range_func, fitness_func) in enumerate(objectives):
            objective_metrics[f"objective_{obj_index}_range"][generation_index] = range_func(population)
            for pop_index, individual in enumerate(population):
                objective_metrics[f"objective_{obj_index}_value"][pop_index, generation_index] = fitness_func(individual)
        
    return genome_metrics, objective_metrics, best_genome


