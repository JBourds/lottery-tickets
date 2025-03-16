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

    @staticmethod
    def from_symbol(symbol: str):
        match symbol.lower():
            case 'hm':
                return Target.HIGH
            case 'lm':
                return Target.LOW
            case 'rand':
                return Target.RANDOM
            case _:
                raise ValueError(f'Unsuported target: {target}')


class Sign(Enum):
    POSITIVE = 0
    NEGATIVE = 1
    BOTH = 2
    FLIP_POSITIVE = 3
    FLIP_NEGATIVE = 4
    FLIP_BOTH = 5
    SAME = 6

    @staticmethod
    def from_symbol(symbol: str):
        match symbol.lower():
            case 's':
                return Sign.SAME
            case 'p':
                return Sign.POSITIVE
            case 'n':
                return Sign.NEGATIVE
            case 'b':
                return Sign.BOTH
            case 'fp':
                return Sign.FLIP_POSITIVE
            case 'fn':
                return Sign.FLIP_NEGATIVE
            case 'fb':
                return Sign.FLIP_BOTH
            case _:
                raise ValueError("Invalid symbol for sign")

# Usage: <lm/hm/rand><% weights to seed>,<sign target>,<scale/set><value>


def get_seeding_rule(seeding_rule: str | None) -> Callable[[List[np.ndarray[float]]], None] | None:
    if seeding_rule is None:
        return None
    match = re.match(
        '^([a-zA-Z]+)(\d+),([npsfb]{1,2}),([a-zA-z]+)([-]?\d+\.*\d*)$', seeding_rule)
    if match is None:
        raise ValueError(
            f'Invalid seeding rule string: {seeding_rule}. Check usage.')
    target, proportion, sign, transform, val = match.groups()
    proportion = float(str("." + proportion))
    if proportion > 1:
        raise ValueError("Cannot have > 100% targeted.")
    val = float(val)
    target = Target.from_symbol(target)
    sign = Sign.from_symbol(sign)
    match transform.lower():
        case 'scale':
            transform = partial(scale_magnitude, factor=val, sign=sign)
        case 'set':
            transform = partial(set_to_constant, constant=val, sign=sign)
        case _:
            raise ValueError(f'Unsuported transform: {transform}')

    # This currently sets the same param for every layer
    return partial(seed_magnitude, targets=target, proportions=proportion, transforms=transform)

# Usage: <lm/hm/rand>-<% weights to seed>
# Will ignore the transformation parameters from the training script


def get_weight_targeting(seeding_rule: str | None) -> WeightsTarget | None:
    if seeding_rule is None:
        return None
    match = re.match('^([a-zA-Z]+)(\d+)', seeding_rule)
    if match is None:
        raise ValueError(
            f'Invalid seeding rule string: {seeding_rule}. Check usage.')
    target, proportion = match.groups()
    proportion = float(str("." + proportion))
    if proportion > 1:
        raise ValueError("Cannot have > 100% targeted.")
    target = Target.from_symbol(target)

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
        raise ValueError(
            "Lists of proportions and strategies must be the same length")
    elif len(model.get_weights()) != len(layer_proportions):
        raise ValueError(
            "Length of lists passed in must match number of model layers with weights")
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
    abs_weights = np.abs(weights)

    # High magnitude
    if target == Target.HIGH:
        threshold = np.sort(abs_weights, axis=None)[-num_weights]
        mag_weights = abs_weights > threshold
    # Low magnitude
    elif target == Target.LOW:
        threshold = np.sort(abs_weights, axis=None)[num_weights - 1]
        mag_weights = abs_weights < threshold
    # Random
    elif target == Target.RANDOM:
        mask = np.ones(abs_weights.size)
        mask[num_weights:] *= 0
        np.random.shuffle(mask)
        mag_weights = np.reshape(mask, abs_weights.shape)
    return mag_weights


def scale_magnitude(
    weights: np.ndarray[float],
    mask: np.ndarray[bool],
    factor: float,
    sign: Sign,
):
    match sign:
        case Sign.SAME:
            weights[mask] *= factor
        case Sign.POSITIVE:
            weights[mask & (weights >= 0)] *= factor
        case Sign.FLIP_POSITIVE:
            weights[mask & (weights >= 0)] *= -factor
        case Sign.NEGATIVE:
            weights[mask & (weights < 0)] *= -factor
        case Sign.FLIP_NEGATIVE:
            weights[mask & (weights < 0)] *= factor
        case Sign.BOTH:
            weights[mask & (weights >= 0)] *= factor
            weights[mask & (weights < 0)] *= -factor
        case Sign.FLIP_BOTH:
            weights[mask & (weights >= 0)] *= -factor
            weights[mask & (weights < 0)] *= factor
        case _:
            raise ValueError(f"Invalid sign: {sign}")


def set_to_constant(
    weights: np.ndarray[float],
    mask: np.ndarray[bool],
    constant: float,
    sign: Sign,
):
    match sign:
        case Sign.SAME:
            weights[mask] = constant
        case Sign.POSITIVE:
            weights[mask & (weights >= 0)] = constant
        case Sign.FLIP_POSITIVE:
            weights[mask & (weights >= 0)] = -constant
        case Sign.NEGATIVE:
            weights[mask & (weights < 0)] = -constant
        case Sign.FLIP_NEGATIVE:
            weights[mask & (weights < 0)] = constant
        case Sign.BOTH:
            weights[mask & (weights >= 0)] = constant
            weights[mask & (weights < 0)] = -constant
        case Sign.FLIP_BOTH:
            weights[mask & (weights >= 0)] = -constant
            weights[mask & (weights < 0)] = constant
        case _:
            raise ValueError(f"Invalid sign: {sign}")

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
