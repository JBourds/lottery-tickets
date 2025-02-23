import numpy as np
import numpy.typing as npt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from typing import Callable, Generator, List, Tuple

from src.harness import architecture as arch
from src.harness import history as hist

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
        i_norm_std = (f_flat - layer_df["li_mean"].iloc[layer]) / layer_df["li_std"].iloc[layer]
         
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
    
    return weight_df

def build_trial_dfs(trial: hist.TrialData, t_num: int, e_num: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    
    return trial_df, layer_df, weight_df

def build_exp_dfs(exp: Generator[hist.TrialData, None, None], e_num: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:      
    trials = []
    layers = []
    weights = []
    print(f"Building dataframes for experiment {e_num}")
    for t_num, t in tqdm(enumerate(exp)): 
        print(f"Trial {t_num}")
        # Dummy line
        t.seed_weights = lambda x: x
        t_df, l_df, w_df = build_trial_dfs(t, t_num=t_num, e_num=e_num)
        trials.append(t_df)
        layers.append(l_df)
        weights.append(w_df)

    trials_df = pd.concat(trials, axis=0, ignore_index=True)
    layers_df = pd.concat(layers, axis=0, ignore_index=True)
    weights_df = pd.concat(weights, axis=0, ignore_index=True)

    return trials_df, layers_df, weights_df

def build_dataframes(epath: str) -> Tuple[pd.DataFrame, ...]:
    experiments = list(hist.get_experiments(epath))
    trials = []
    layers = []
    weights = []
    print(f"Building dataframes for experiments at {epath}")
    for e_num, e in tqdm(enumerate(experiments)): 
        if len(trials) >= 2:
            break
        print(f"Experiment {e_num}")
        t_df, l_df, w_df = build_exp_dfs(e, e_num)
        trials.append(t_df)
        layers.append(l_df)
        weights.append(w_df)

    trials_df = pd.concat(trials, axis=0, ignore_index=True)
    layers_df = pd.concat(layers, axis=0, ignore_index=True)
    weights_df = pd.concat(weights, axis=0, ignore_index=True)

    return trials_df, layers_df, weights_df

def merge_dfs(
    trials_df: pd.DataFrame,
    layers_df: pd.DataFrame,
    weights_df: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["e_num", "t_num", "l_num", "w_num"]
    tl_df = pd.merge(trials_df, layers_df, on=keys[:2])
    tlw_df = pd.merge(tl_df, weights_df, on=keys[:3], how="outer")
    tlw_df = pd.get_dummies(tlw_df, columns=["arch"])
    tlw_df = pd.get_dummies(tlw_df, columns=["dataset"])
    drop_cols = keys + ["step", "seed"]
    tlw_df.drop(columns=drop_cols, inplace=True)
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