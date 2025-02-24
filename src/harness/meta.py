from copy import copy as shallowcopy
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from tensorflow import keras
from typing import Callable, List, Tuple

def create_meta(shape: Tuple[int, ...], depth: int, width: int) -> keras.Model:
    model = keras.Sequential([keras.Input(shape=shape)])
    for _ in range(depth):
        model.add(keras.layers.Dense(width, "relu"))
    model.add(keras.layers.Dense(1, "sigmoid"))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
             loss=tf.keras.losses.BinaryCrossentropy(),
             metrics=["accuracy"])
    return model


def make_meta_mask(
    meta: keras.Model,
    make_x: Callable[[str, str, keras.Model, List[npt.NDArray]], npt.NDArray],
    architecture: str,
    dataset: str,
    steps: int,
) -> Tuple[List[npt.NDArray], List[float]]:
    a = arch.Architecture(architecture, dataset)
    _, val_X, _, val_Y = a.load_data()
    model = a.get_model_constructor()()
    original_weights = copy.deepcopy(model.get_weights())
    model.compile(optimizer="Adam", loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])
    masks = [np.ones_like(w) for w in model.get_weights()]
    
    def update_masks(mask_pred: npt.NDArray) -> List[npt.NDArray]:
        start = 0
        end = 0
        new_masks = []
        nonlocal masks
        for m in masks:
            end += m.size
            new_m = np.reshape(mask_pred[start:end], m.shape)
            new_masks.append(new_m)
            start = end
        return masks
            
    accuracies = []
    for step in range(steps):
        # Get validation accuracy
        _, accuracy = model.evaluate(val_X, val_Y)
        accuracies.append(accuracy)
        print(f"Step {step} accuracy: {accuracy:.2%}")
        # Extract features
        X = make_x(architecture, model, masks)
        # Predict and replace existing mask
        mask_pred = meta.predict(X, batch_size=2**20)
        masks = update_masks(mask_pred)
        model.set_weights([w * m for w, m in zip(original_weights, masks)])
        
    return masks, accuracies


def make_x(
    architecture: str,
    model: keras.Model,
    masks: List[npt.NDArray],
) -> npt.NDArray:
    # Layer features:
    # i_features = ["l_sparsity", "l_rel_size", "li_prop_positive", "wi_std", "wi_perc", "wi_synflow", "wi_sign", "dense", "bias", "conv", "output"]
    nparams = sum(map(np.size, masks))
    nfeatures = 11
    features = np.zeros((nparams, nfeatures))
    
    # Helper functions to add the unrolled weight values and
    # scalar layer values to the feature matrix
    n = 0
    def add_layer_features(layer_values: List[float]):
        nonlocal n
        start = 0
        end = 0
        for v, size in zip(layer_values, map(np.size, masks)):
            end += size
            features[start:end, n] = v
            start = end
        n += 1
        
    def add_weight_features(weight_features: List[npt.NDArray]):
        nonlocal n
        start = 0
        end = 0
        for v in weight_features:
            end += v.size
            features[start:end, n] = np.ravel(v)
            start = end
        n += 1
    
    # Make a separate copy to compute synflow for
    masked_weights = [w * m for w, m in zip(model.get_weights(), masks)]
    masked_model = shallowcopy(model)
    masked_model.set_weights(masked_weights)
    synflow_scores = [np.reshape(scores, -1) for scores in compute_synflow_per_weight(masked_model)]
    
    # Mask features
    sparsities = [np.count_nonzero(m) / np.size(m) for m in masks]
    rel_size = [np.size(m) / nparams for m in masks]
    prop_pos = [np.count_nonzero(w >= 0) for w in masks]
    
    # Layer type
    layer_ohe = arch.Architecture.ohe_layer_types(architecture)
    for values in [sparsities, rel_size, prop_pos]:
        add_layer_features(values)
    
    # Weight features
    l_std = [np.std(w) for w in masked_weights]
    l_mean = [np.mean(w) for w in masked_weights]
    l_sorted = [np.sort(np.ravel(w)) for w in masked_weights]
    
    w_std = [(w - l_mean) / l_std for w, l_mean, l_std in zip(l_std, l_mean, masked_weights)]
    w_sign = [np.sign(w) for w in masked_weights]
    num_nonzero = sum(map(np.count_nonzero, masks))
    num_zero = nparams - num_nonzero
    w_perc = np.array([
        np.argmax(np.ravel(v) < v_sorted) - num_zero 
        for v, v_sorted in zip(masked_weights, l_sorted)]
    ) / num_nonzero
    
    flat_masks = [np.ravel(m) for m in masks]
    for values in [w_std, w_perc, synflow_scores, w_sign]:
        add_weight_features(values)
    
    for values in [layer_ohe[:, i] for i in range(layer_ohe.shape[1])]:
        add_layer_features(values)
        
    return features
