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
from typing import Any, Callable, Dict, List, Optional, Tuple

# Type for a function which performs some kind of initialization
# Strategy on weights given some boolean mask to apply it to
WeightsTarget = Callable[
    [
        np.ndarray,
        float,
    ], 
    np.ndarray[int]
]
InitStrategy = Callable[
    [
        np.ndarray[np.float64],
        np.ndarray[bool], 
        WeightsTarget,
        Callable[[Any], None],
    ], 
    None
]
WeightsTransform = Callable[
    [
        np.ndarray[float],
        np.ndarray[bool],
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
    elif any(map(lambda p: p > 1 or p < 0, layer_proportions)):
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
    mask: List[np.ndarray[bool]], 
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
    num_weights = int(weights.size * proportion)
    weights = np.abs(weights)
    
    # High magnitude
    if target == Target.HIGH:
        threshold = np.sort(weights, axis=None)[-num_weights]
        return weights > threshold
    # Low magnitude
    elif target == Target.LOW:
        threshold = np.sort(weights, axis=None)[num_weights - 1]
        return weights < threshold
    # Random
    elif target == Target.RANDOM:
        mask = np.ones(weights.size)
        mask[num_weights:] *= 0
        np.random.shuffle(mask)
        return np.reshape(mask, weights.shape)
    
def scale_magnitude(
    weights: np.ndarray[float],
    mask: np.ndarray[bool],
    factor: float,
):
    weights[mask] *= factor
    
def set_to_constant(
    weights: np.ndarray[float],
    mask: np.ndarray[bool],
    constant: float,
):
    weights[mask] = constant
    
# Init strategies to modify weights in place
def seed_magnitude(
    weights: List[np.ndarray[float]],
    proportions: List[float] | float,
    targets: List[Target] | Target = Target.HIGH,
    transforms: List[WeightsTransform] | WeightsTransform = lambda x: x,
):
    if type(proportions) != list:
        proportions = [proportions] * len(weights)
    if type(targets) != list:
        targets = [targets] * len(weights)
    if type(transforms) != list:
        transforms = [transforms] * len(weights)
        
    for weights, prop, target, transform in zip(weights, proportions, targets, transforms):
        mask = target_magnitude(weights, prop, target)
        transform(weights, mask)
     
