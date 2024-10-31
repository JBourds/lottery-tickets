"""
seeding.py

Module containing function definitions used for "Seeding"
models with various types of initializations (e.g., 
10% high magnitude initializations) along
with callbacks for tracking metrics over the course of
training.

Author: Jordan Bourdeau
"""

from enum import Enum
import numpy as np
from tensorflow import keras
from typing import Callable, List, Optional

# Type for a function which performs some kind of initialization
# Strategy on weights given some boolean mask to apply it to
WeightsTarget = Callable[
    [
        weights: np.ndarray, 
        proportion: float,
    ], 
    np.ndarray[int]
]
InitStrategy = Callable[
    [
        weights: np.ndarray,
        mask: np.ndarray, 
        target_rule: WeightsTarget,
        transform: Callable[[Any], None],
    ], 
    None
]

class Target(Enum):
    HIGH = 0
    LOW = 1
    RANDOM = 2

def init_seed(
    model: keras.Model,
    layer_proportions: List[float],
    init_strategies: List[InitStrategy],
) -> List[np.ndarray[bool]]:
    if len(layer_proportions) != len(init_strategies):
        raise ValueError("Lists of proportions and strategies must be the same length")
    elif len(model.get_weights()) != len(layer_proportions):
        raise ValueError("Length of lists passed in must match number of model layers with weights")
    elif any(map(lambda p: p > 1 or p < 0, layer_proportions):
        raise ValueError("All proportions must be between 0 and 1")
    
    new_weights = model.get_weights()
    mask = [
        init(w, p) for init, w, p 
        in zip(init_strategies, new_weights, layer_proportions)
    ]
    model.set_weights(new_weights)
    return mask

def init_callback(
    model: keras.Model, 
    mask: List[np.ndarray[mask]], 
    metrics: List[Callable[[Tuple[str, np.ndarray[bool]]], Any]],
) -> Dict:
    return {
        name: [metric(w[i]) for w, i in zip(model.get_weights(), mask)]
        for name, metric in metrics
    }

# Target rules for selecting weights
def target_magnitude(
    weights: np.ndarray[float],
    proportion: float,
    target: Target,
) -> np.ndarray[int]:
    num_weights = weights.size * proportion
    match target:
        Target.HIGH:
            threshold = np.sort(weights)[-num_weights]
            return weights > threshold
        Target.LOW:
            threshold = np.sort(weights)[num_weights - 1]
            return weights < threshold
        Target.RANDOM:
            mask = np.ones(weights.size)
            mask[num_weights:] *= 0
            np.random.shuffle(mask)
            return np.reshape(mask, weights.shape)
    
# Init strategies to modify weights in place
def seed_magnitude(
    weights: np.ndarray[float],
    proportion: float,
    target: Target = Target.HIGH,
    transform: Callable[[Any], None] = lambda x: x,
):
    mask = target_magnitude(weights, proportion, target)
    transform(weights[mask])
     
