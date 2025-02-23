import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense
from typing import List, Dict

from . import get_layer_metric_array

# NOTE: Set TF backend to float64 or else a lot of small synflow scores will be lost
def compute_synflow_per_weight(net: keras.Model) -> List[tf.Tensor]:
        
    def linearize() -> Dict[str, tf.Tensor]:
        signs = {}
        for layer in net.trainable_variables:
            if hasattr(layer, 'kernel'):
                signs[layer.name] = tf.sign(layer.kernel)
                layer.kernel.assign(tf.abs(layer.kernel))
        return signs

    def nonlinearize(signs: Dict[str, tf.Tensor]) -> None:
        for layer in net.trainable_variables:
            if hasattr(layer, 'kernel'):
                layer.kernel.assign(layer.kernel * signs[layer.name])
    
    signs = linearize()

    input_dim = [1] + list(net.input_shape[1:])
    inputs = tf.ones(input_dim, dtype=tf.float64)
    with tf.GradientTape() as tape:
        output = net(inputs)
        loss = tf.reduce_sum(output)
    grads = tape.gradient(loss, net.trainable_variables)
    scores = [tf.abs(l * g) for l, g in zip(net.trainable_variables, grads)]

    nonlinearize(signs)
    
    return scores

