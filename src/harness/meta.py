from copy import copy as shallowcopy
import copy
import numpy as np
import numpy.typing as npt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from typing import Callable, Dict, List, Tuple
from sklearn.model_selection import train_test_split

from src.harness import architecture as arch
from src.metrics import features as f
from src.metrics.synflow import compute_synflow_per_weight


def create_meta(shape: Tuple[int, ...], depth: int, width: int) -> keras.Model:
    """
    Create the meta model with a specified depth and width for each layer.
    """
    model = keras.Sequential([keras.Input(shape=shape)])
    for _ in range(depth):
        model.add(keras.layers.Dense(width, "relu"))
    model.add(keras.layers.Dense(1, "sigmoid"))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=["accuracy"])
    return model


def train_meta(
    meta_model: keras.Model,
    X: npt.NDArray,
    Y: npt.NDArray,
    epochs: int = 2,
    batch_size: int = 256,
    val_prop: float = 0.2,
) -> Dict:
    """
    Function which trains the meta model and analyzes the types of mistakes
    it makes.
    """
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)
    history = meta_model.fit(X_train, Y_train, epochs=epochs,
                             batch_size=batch_size, validation_split=val_prop, shuffle=True)
    loss, accuracy = meta_model.evaluate(X_test, Y_test)
    return history, loss, accuracy


def make_meta_mask(
    predict: Callable[[npt.NDArray], npt.NDArray],
    make_x: Callable[[str, str, keras.Model, List[npt.NDArray]], npt.NDArray],
    architecture: str,
    dataset: str,
    epochs: int,
    train_steps: int,
    features: List[str],


) -> Tuple[List[npt.NDArray], List[float]]:
    """
    Function which simulates training but only relies on the predicted masks
    from the meta model.

    @param predict: Function which takes X array as input and outputs a 0 or 1
        for each feature vector corresponding to each neuron.
    @param make_x: Function that produces the input vector for the meta model
        based on the keras model.
    @param architecture: String for the architecture being used.
    @param dataset: String for the dataset being used.
    @param epochs: Number of masking iterations to use.
    @param train_steps: Number of training steps to use.
    @param features: Feature names to use.

    @returns: Final masks and accuracies.
    """
    a = arch.Architecture(architecture, dataset)
    _, X_test, _, Y_test = a.load_data()
    model = a.get_model_constructor()()
    original_weights = copy.deepcopy(model.get_weights())
    model.compile(optimizer="Adam", loss=keras.losses.CategoricalCrossentropy(
    ), metrics=["accuracy"])
    masks = [np.ones_like(w) for w in model.get_weights()]
    layer_names = list(map(lambda x: x.lower(), a.layer_names))

    def get_sparsities() -> List[float]:
        return [np.count_nonzero(m) / m.size for m in masks]

    def update_masks(mask_pred: npt.NDArray):
        start = 0
        end = 0
        nonlocal masks
        for layer, (mask, name) in enumerate(zip(masks, layer_names), start=1):
            end += mask.size
            # And with existing mask to not allow previously masked
            # weights to become unmasked by the xor
            to_prune = np.logical_and(mask_pred[start:end].astype(
                bool), mask.astype(bool).ravel()).astype(int)
            # Output & convolutional layers, prune at half the rate
            if "output" in name or "conv" in name:
                to_prune *= np.random.randint(low=0,
                                              high=2, size=to_prune.shape)
            # Because to_prune can only have 1s where there is also a mask,
            # this works as a bit flip only where there are 1s
            mask -= np.reshape(to_prune, mask.shape)
            start = end

    accuracies = []
    for step in range(epochs):
        # Get validation accuracy
        _, accuracy = model.evaluate(X_test, Y_test)
        accuracies.append(accuracy)
        print(
            f"Step {step} accuracy: {accuracy:.2%}, sparsities: {get_sparsities()}")
        # Extract features
        X = make_x(architecture, dataset, model, masks, features, train_steps)
        # Predict and replace existing mask
        mask_pred = predict(X)
        update_masks(mask_pred)
        model.set_weights([w * m for w, m in zip(original_weights, masks)])

    return masks, accuracies


def make_x(
    architecture: str,
    dataset: str,
    model: keras.Model,
    masks: List[npt.NDArray],
    features: List[str],
    train_steps: int = 10,
    batch_size: int = 256,
) -> npt.NDArray:
    """
    Function that creates the feature matrix for the meta model.
    Creates a DF with all features in it, then selects the subset.

    @param architecture: Architecture string used to get information
        about model layers.
    @param dataset: Dataset string used to signify which dataset was used.
    @param model: Keras model.
    @param masks: Current masks (used to compute metrics from masked model).
    @param features: List of features to use.

    @returns: Feature matrix used by the meta model.
    """
    layer_df = f.build_layer_df(architecture, model.get_weights(), [], masks)
    architecture = arch.Architecture(architecture, dataset)
    # Use masks twice here since we aren't generating any labels but just need the shape
    weight_df = f.build_weight_df(
        layer_df, architecture, model.get_weights(), [], masks, [], training=False)
    trained_weight_df = f.build_weight_df_with_training(
        layer_df, architecture, model.get_weights(), masks, train_steps, batch_size, training=False) if train_steps > 0 else None

    df = pd.merge(layer_df, weight_df, on=["l_num"], how="inner")
    if trained_weight_df is not None:
        df = pd.merge(df, trained_weight_df, on=[
                      "l_num", "w_num"], how="inner")
    X = df[features].to_numpy()

    return X
