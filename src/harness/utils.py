"""
utils.py

File containing utility functions.
"""

import numpy as np
import tensorflow as tf

from src.harness.constants import Constants as C

def count_nonzero_parameters(model: tf.keras.Model):
    """
    Print summary for the number of nonzero parameters in the model.
    """
    model_sum_params: int = 0
    weights: list[np.ndarray] = model.trainable_weights
    for idx in range(len(weights))[::2]:
        layer_number: int = int(idx / 2)
        synapses: np.ndarray = weights[idx]
        nonzero_synapses: int = tf.math.count_nonzero(synapses, axis=None).numpy()
        neurons: np.array = weights[idx + 1]
        nonzero_neurons: int = tf.math.count_nonzero(neurons, axis=None).numpy()

        print(f'Nonzero parameters in layer {layer_number} synapses:', nonzero_synapses)
        print(f'Nonzero parameters in layer {layer_number} neurons:', nonzero_neurons)
        
        model_sum_params += nonzero_synapses + nonzero_neurons
    
    print(f'Total nonzero parameters: {model_sum_params}')