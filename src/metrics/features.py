import numpy as np
import numpy.typing as npt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from typing import Callable, Generator, List, Optional, Tuple

from src.harness import architecture as arch
from src.harness import history as hist
from src.metrics.synflow import *

def build_layer_df(
    architecture: str, 
    initial: List[npt.NDArray], 
    final: List[npt.NDArray], 
    masks: List[npt.NDArray],
) -> pd.DataFrame:
    layer_features = {
        "l_num": [],
        "l_size": [],
        "l_rel_size": [],
        "l_sparsity": [],
        "lf_mean": [],
        "lf_std": [],
        "lf_prop_positive": [],
        "li_mean": [],
        "li_std": [],
        "li_prop_positive": [],
    }
     # Loop over layers, vectorize calculations where possible
    print("Creating layer features")
    total_size = sum(map(np.size, masks))
    for index, (iw, fw, m) in tqdm(enumerate(zip(initial, final, masks))):
        print(f"Layer {index}")
        mask = m.astype(bool)
        fw_filtered = fw[mask]
        iw_filtered = iw[mask]
        layer_features["l_num"].append(index) 
        layer_features["l_size"].append(m.size) 
        layer_features["l_rel_size"].append(m.size / total_size) 
        layer_features["l_sparsity"].append(np.mean(m))
        layer_features["lf_mean"].append(np.mean(fw_filtered))
        layer_features["lf_std"].append(np.std(fw_filtered)) 
        layer_features["lf_prop_positive"].append(np.mean(fw_filtered >= 0))  
        layer_features["li_mean"].append(np.mean(iw_filtered))  
        layer_features["li_std"].append(np.std(iw_filtered)) 
        layer_features["li_prop_positive"].append(np.mean(iw_filtered >= 0)) 
    layer_ohe = arch.Architecture.ohe_layer_types(architecture)
    for index, name in enumerate(arch.Architecture.LAYER_TYPES):
        layer_features[name] = [ohe[index] for ohe in layer_ohe]
    layer_df = pd.DataFrame(layer_features)
    
    return layer_df

def normalize(x: pd.Series) -> pd.Series:
    return (x - x.min()) / (x.max() - x.min())

def get_train_one_step() -> Callable:
    @tf.function
    def train_one_step(
        model: tf.keras.Model,
        masks: List[tf.Tensor],
        inputs: tf.Tensor,
        labels: tf.Tensor,
        optimizer: tf.keras.optimizers.Optimizer,
        loss_fn: tf.keras.metrics.Metric,
    ) -> float:
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = loss_fn(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_weights)
        grad_mask_mul = []
        for grad_layer, mask in zip(gradients, masks):
            grad_mask_mul.append(tf.math.multiply(grad_layer, mask))
        optimizer.apply_gradients(
            zip(grad_mask_mul, model.trainable_weights))
        
        return loss

    return train_one_step

def build_weight_df_with_training(
    layer_df: pd.DataFrame,
    architecture: arch.Architecture,
    weights: List[npt.NDArray[np.float32]],
    masks: List[npt.NDArray[np.float32]],
    n: int, 
    batch_size: int,
) -> pd.DataFrame:
    # Training
    model = architecture.get_model_constructor()()
    model.set_weights([w * m for w, m in zip(weights, masks)])
    X_train, _, Y_train, _ = architecture.load_data()
    masks = list(map(lambda x: tf.convert_to_tensor(x, dtype=tf.float64), masks))
    X_train = tf.random.shuffle(X_train, seed=0)
    Y_train = tf.random.shuffle(Y_train, seed=0)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    @tf.function
    def do_train_steps() -> List[float]:
        losses = []
        for i in tqdm(range(n)):
            start = i * batch_size
            stop = start + batch_size
            inputs, labels = X_train[start:stop], Y_train[start:stop]
            train_one_step = get_train_one_step()
            losses.append(train_one_step(model, masks, inputs, labels, optimizer, loss_fn))
        return losses
    
    print("Training...")
    losses = do_train_steps()
    trained = model.get_weights()
    
    # Compute features
    weight_features_list = []
    start_idx = 0
    print("Making weight features with training")
    num_weights = sum(map(np.size, model.get_weights()))
    shape = [1] + list(model.input_shape[1:])
    t_synflow = compute_synflow_per_weight(model)
    
    # Duplicate features for initial vs. final weights
    for layer, (tw, m) in tqdm(enumerate(zip(trained, masks))):
        print(f"Layer {layer}")
        mask = m.numpy().astype(bool).ravel()
        num_params = len(mask)
        num_nonzero = np.count_nonzero(mask)
        num_zero = num_params - num_nonzero
        # Precompute values for this layer
        layer_num = np.full(num_params, layer, dtype=np.int8)
        weight_nums = np.arange(num_params, dtype=np.int32)

        # Don't preserve initial weight if a weight was masked
        t_flat = tw.flatten() * mask
        t_sorted = np.sort(t_flat)
        t_sign = np.sign(t_flat)
        t_mag = np.abs(t_flat, dtype=np.float32)
        t_perc = np.array([np.argmax(v < t_sorted) - num_zero for v in t_flat]) / num_nonzero
        # Use std from initial weights assuming we did small amounts of training
        t_norm_std = (t_flat - layer_df["li_mean"].iloc[layer]) / layer_df["li_std"].iloc[layer]
         
        # Create a dictionary for weight features for this layer
        layer_weight_features = {
            "l_num": layer_num,
            "w_num": weight_nums,
            f"wt{n}_sign": t_sign,
            f"wt{n}_val": t_flat,
            f"wt{n}_mag": t_mag,
            f"wt{n}_perc": t_perc.astype(np.float32),
            f"wt{n}_std": t_norm_std.astype(np.float32),
            f"wt{n}_synflow": t_synflow[layer].numpy().flatten(),
        }
        
        weight_features_list.append(pd.DataFrame(layer_weight_features))

    weight_df = pd.concat(weight_features_list, axis=0, ignore_index=True)
    
    # Bonus features
    keys = ["l_num"]
    weight_df[f"norm_wt{n}_mag"] = weight_df.groupby(keys)[f"wt{n}_mag"].transform(normalize)
    weight_df[f"norm_wt{n}_synflow"] = weight_df.groupby(keys)[f"wt{n}_synflow"].transform(normalize)
    weight_df.fillna(0, inplace=True)
    
    return weight_df

def build_weight_df(
    layer_df: pd.DataFrame,
    architecture: arch.Architecture,
    initial: List[npt.NDArray],
    final: List[npt.NDArray],
    masks: List[npt.NDArray],
) -> pd.DataFrame:
    # Prepare weight features: No need for large weight_features array, use a list of dicts
    weight_features_list = []
    start_idx = 0

    print("Making weight features")
    model = architecture.get_model_constructor()()
    num_weights = sum(map(np.size, model.get_weights()))
    shape = [1] + list(model.input_shape[1:])
    
    # Computed across the whole model
    model.set_weights([w * m for w, m in zip(initial, masks)])
    i_synflow = compute_synflow_per_weight(model)
    model.set_weights([w * m for w, m in zip(final, masks)])
    f_synflow = compute_synflow_per_weight(model)
    
    # Duplicate features for initial vs. final weights
    for layer, (iw, fw, m) in tqdm(enumerate(zip(initial, final, masks))):
        print(f"Layer {layer}")
        mask = m.astype(bool).ravel()
        num_params = len(mask)
        num_nonzero = np.count_nonzero(mask)
        num_zero = num_params - num_nonzero
        # Precompute values for this layer
        layer_num = np.full(num_params, layer, dtype=np.int8)
        weight_nums = np.arange(num_params, dtype=np.int32)
        
        f_flat = fw.flatten()
        f_sorted = np.sort(f_flat)
        f_sign = np.sign(f_flat)
        f_mag = np.abs(f_flat, dtype=np.float32)
        f_perc = np.array([np.argmax(v < f_sorted) - num_zero for v in f_flat]) / num_nonzero
        f_norm_std = (f_flat - layer_df["lf_mean"].iloc[layer]) / layer_df["lf_std"].iloc[layer]
        
        
        i_flat = iw.flatten()
        i_sorted = np.sort(i_flat)
        i_sign = np.sign(i_flat)
        i_mag = np.abs(i_flat, dtype=np.float32)
        i_perc = np.array([np.argmax(v < i_sorted) - num_zero for v in i_flat]) / num_nonzero
        i_norm_std = (i_flat - layer_df["li_mean"].iloc[layer]) / layer_df["li_std"].iloc[layer]
         
        # Create a dictionary for weight features for this layer
        layer_weight_features = {
            "l_num": layer_num,
            "w_num": weight_nums,
            "wf_sign": f_sign,
            "wi_sign": i_sign,
            "wf_val": f_flat,
            "wi_val": i_flat,
            "wf_mag": f_mag,
            "wi_mag": i_mag,
            "wf_perc": f_perc.astype(np.float32),
            "wi_perc": i_perc.astype(np.float32),
            "wf_std": f_norm_std.astype(np.float32),
            "wi_std": i_norm_std.astype(np.float32),
            "w_mask": mask,
            "wf_synflow": f_synflow[layer].numpy().flatten(),
            "wi_synflow": i_synflow[layer].numpy().flatten(),
        }
        
        weight_features_list.append(pd.DataFrame(layer_weight_features))

    weight_df = pd.concat(weight_features_list, axis=0, ignore_index=True)

    # Bonus features
    keys = ["l_num"]
    weight_df["norm_wi_mag"] = weight_df.groupby(keys)["wi_mag"].transform(normalize)
    weight_df["norm_wi_synflow"] = weight_df.groupby(keys)["wi_synflow"].transform(normalize)
    weight_df.fillna(0, fillna=True)
    
    return weight_df

def build_trial_dfs(
    trial: hist.TrialData, 
    t_num: int, 
    e_num: int,
    train_steps: int,
    batch_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    # Trial DataFrame
    total_size = sum(map(np.size, trial.masks))
    sparsity = sum(map(np.sum, trial.masks)) / total_size
    trial_df = pd.DataFrame({
        "e_num": [e_num],
        "t_num": [t_num],
        "seed": [trial.random_seed],
        "step": [trial.pruning_step],
        "arch": [trial.architecture],
        "dataset": [trial.dataset],
        "sparsity": [sparsity],
        "size": [total_size],
    })
    trial_df["sparsity"] = trial_df["sparsity"].astype("float32")
    
    def add_labels(df: pd.DataFrame):
        df["e_num"] = e_num
        df["e_num"] = df["e_num"].astype("uint8")
        df["t_num"] = t_num
        df["t_num"] = df["t_num"].astype("uint8")
    
    # Layer feature computation
    layer_df = build_layer_df(trial.architecture, trial.initial_weights, trial.final_weights, trial.masks)
    add_labels(layer_df)
    
    architecture = arch.Architecture(trial.architecture, trial.dataset)
    weight_df = build_weight_df(layer_df, architecture, trial.initial_weights, trial.final_weights, trial.masks)
    add_labels(weight_df)
    
    if train_steps > 0:
        trained_weight_df = build_weight_df_with_training(
            layer_df, 
            architecture, 
            trial.initial_weights,
            trial.masks,
            train_steps,
            batch_size,
        )
        add_labels(trained_weight_df)
    else:
        trained_weight_df = None
    
    # Correct the class imbalance- merge with trained weights later
    weight_df = correct_class_imbalance(weight_df)
    
    return trial_df, layer_df, weight_df, trained_weight_df

def build_exp_dfs(
    exp: Generator[hist.TrialData, None, None], 
    e_num: int,
    train_steps, 
    batch_size,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:      
    trials = []
    layers = []
    weights = []
    trained_weights = []
    print(f"Building dataframes for experiment {e_num}")
    for t_num, t in tqdm(enumerate(exp)): 
        print(f"Trial {t_num}")
        # Dummy line
        t.seed_weights = lambda x: x
        t_df, l_df, w_df, tw_df = build_trial_dfs(
            t, 
            t_num=t_num, 
            e_num=e_num, 
            train_steps=train_steps, 
            batch_size=batch_size,
        )
        trials.append(t_df)
        layers.append(l_df)
        weights.append(w_df)
        if tw_df is not None:
            trained_weights.append(tw_df)

    trials_df = pd.concat(trials, axis=0, ignore_index=True)
    layers_df = pd.concat(layers, axis=0, ignore_index=True)
    weights_df = pd.concat(weights, axis=0, ignore_index=True)
    trained_weights_df = pd.concat(trained_weights, axis=0, ignore_index=True) if trained_weights else None

    return trials_df, layers_df, weights_df, trained_weights_df

def build_dataframes(
    epath: str, 
    train_steps: int = 0, 
    batch_size: int = 64,
) -> Tuple[pd.DataFrame, ...]:
    experiments = list(hist.get_experiments(epath))
    trials = []
    layers = []
    weights = []
    trained_weights = []
    print(f"Building dataframes for experiments at {epath}")
    for e_num, e in tqdm(enumerate(experiments)): 
        # TODO: Remove to get full dataset
        if len(trials) >= 2:
            break
        print(f"Experiment {e_num}")
        t_df, l_df, w_df, tw_df = build_exp_dfs(e, e_num, train_steps, batch_size)
        trials.append(t_df)
        layers.append(l_df)
        weights.append(w_df)
        if tw_df is not None:
            trained_weights.append(tw_df)

    trials_df = pd.concat(trials, axis=0, ignore_index=True)
    layers_df = pd.concat(layers, axis=0, ignore_index=True)
    weights_df = pd.concat(weights, axis=0, ignore_index=True)
    trained_weights_df = pd.concat(trained_weights, axis=0, ignore_index=True) if trained_weights else None

    return trials_df, layers_df, weights_df, trained_weights_df

def merge_dfs(
    trials_df: pd.DataFrame,
    layers_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    trained_weights_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    keys = ["e_num", "t_num", "l_num", "w_num"]
    tl_df = pd.merge(trials_df, layers_df, on=keys[:2])
    tlw_df = pd.merge(tl_df, weights_df, on=keys[:3], how="outer")
    if trained_weights_df is not None:
        tlw_df = pd.merge(tlw_df, trained_weights_df, on=keys, how="inner")
    tlw_df = pd.get_dummies(tlw_df, columns=["arch"])
    tlw_df = pd.get_dummies(tlw_df, columns=["dataset"])
    # drop_cols = keys + ["step", "seed"]
    # tlw_df.drop(columns=drop_cols, inplace=True)
    return tlw_df

# Correct for class imblance by trimming groups to the smallest
# size. Can afford to do this because we have so many data points
def correct_class_imbalance(wdf: pd.DataFrame) -> pd.DataFrame:
    smallest_group_size = min(wdf["w_mask"].value_counts())
    dfs = []
    for _, group in wdf.groupby("w_mask"):
        dfs.append(group.sample(smallest_group_size, replace=False))
    return pd.concat(dfs)

def featurize_db(
    tlw_df: pd.DataFrame,
    features: List[str] = [],
) -> Tuple[npt.NDArray, npt.NDArray]: 
    if not features:
        raise ValueError("No features provided.")
    Y, X = tlw_df["w_mask"], tlw_df[features]
    return X.to_numpy().astype(np.float32), Y.to_numpy()
