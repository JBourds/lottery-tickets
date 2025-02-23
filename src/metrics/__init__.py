import tensorflow as tf
from tensorflow import keras
from typing import Callable, List

def get_layer_metric_array(model: keras.Model, metric: Callable) -> List[tf.Tensor]:
    metric_array = []
    for layer in model.layers:
        if isinstance(layer, Conv2D) or isinstance(layer, Dense) and layer.activation == tf.keras.activations.linear:
            metric_array.append(metric(layer))
        else:
            metric_array.append([])
    return metric_array
