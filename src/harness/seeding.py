"""
py

Module containing function definitions used for "Seeding"
models with various types of initializations (e.g., 
10% high magnitude initializations) along
with callbacks for tracking metrics over the course of
training.

Author: Jordan Bourdeau
"""

from enum import Enum
from functools import partial
import numpy as np
import re
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

# Usage: <lm/hm/rand>-<% weights to seed>-<scale/set>-<value>
def get_seeding_rule(seeding_rule: str | None) -> Callable[[List[np.ndarray[float]]], None] | None:
    if seeding_rule is None:
        return None
    match = re.match('^([a-zA-Z]+)(\d{1,3}),([a-zA-z]+)([-]?\d+\.*\d*)$', seeding_rule)
    if match is None:
        raise ValueError(f'Invalid seeding rule string: {seeding_rule}. Check usage.')
    target, proportion, transform, val = match.groups()
    proportion = float(proportion) / 100
    val = float(val)
    match target.lower():
        case 'hm':
            target = Target.HIGH
        case 'lm':
            target = Target.LOW
        case 'rand':
            target = Target.RANDOM
        case _:
            raise ValueError(f'Unsuported target: {target}')
    match transform.lower():
        case 'scale':
            transform = partial(scale_magnitude, factor=val)
        case 'set':
            transform = partial(set_to_constant, constant=val)
        case _:
            raise ValueError(f'Unsuported transform: {transform}')
    
    # This currently sets the same param for every layer
    return partial(seed_magnitude, targets=target, proportions=proportion, transforms=transform)

# Usage: <lm/hm/rand>-<% weights to seed>
# Will ignore the transformation parameters from the training script
def get_weight_targeting(seeding_rule: str | None) -> WeightsTarget | None:
    if seeding_rule is None:
        return None
    match = re.match('^([a-zA-Z]+)(\d{1,3})', seeding_rule)
    if match is None:
        raise ValueError(f'Invalid seeding rule string: {seeding_rule}. Check usage.')
    target, proportion = match.groups()
    proportion = float(proportion) / 100
    match target.lower():
        case 'hm':
            target = Target.HIGH
        case 'lm':
            target = Target.LOW
        case 'rand':
            target = Target.RANDOM
        case _:
            raise ValueError(f'Unsuported target: {target}')
    
    # This currently sets the same param for every layer
    return partial(
        target_magnitude,
        proportion=proportion,
        target=target,
    )
    
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
    transforms: List[WeightsTransform] | WeightsTransform = None
):
    if type(proportions) != list:
        proportions = [proportions] * len(weights)
    if type(targets) != list:
        targets = [targets] * len(weights)
    if type(transforms) != list:
        transforms = [transforms] * len(weights)
        
    for weights, prop, target, transform in zip(weights, proportions, targets, transforms):
        mask = target_magnitude(weights, prop, target)
        if transform is not None:
            transform(weights, mask)
     
