"""
paths.py

Utilities for building paths to store data.

Created By: Jordan Bourdeau
Date: 3/17/24
"""

import os

from src.harness import constants as C

def get_model_directory(
    seed: int, 
    pruning_step: int, 
    masks: bool = False, 
    initial: bool = False, 
    trial_data: bool = False, 
    experiment_data: bool = False,
    parent_directory: str = C.MODELS_DIRECTORY,
    experiment_prefix: str = C.EXPERIMENT_PREFIX,
    ) -> str:
    """
    Function used to retrieve a model's directory.

    Args:
        seed (int): Random seed the model was trained using.
        pruning_step (int): Integer value for the number of pruning steps which had been completed for the model.
        masks (bool): Boolean for whether the model masks are being retrieved or not. Default is False.
        initial (bool): Boolean for whether to use the initial directory. Default is False.
        trial_data (bool): Boolean for whether to get the directory to put the round data in. Default is False.
        experiment_data (bool): Boolean for whether to get the directory to put the experiment data in. Default is False.
        parent_directory (str): Parent directory of the model. Default is C.MODELS_DIRECTORY.

    Returns:
        str: Model directory.
    """
    output_directory: str = os.path.join(parent_directory, f'{experiment_prefix}{seed}')
    if not experiment_data:
        trial_directory: str = initial_dir(output_directory) if initial else trial_dir(output_directory, pruning_step)
        target_directory: str = trial_directory if trial_data else mask_dir(trial_directory) if masks else weights_dir(trial_directory)
    return target_directory

def get_model_filepath(seed: int, pruning_step: int, masks: bool = False, initial: bool = False) -> str:
    """
    Function used to retrieve a model's file path.

    Args:
        seed (int): Random seed the model was trained using.
        pruning_step (int): Integer value for the number of pruning steps which had been completed for the model.
        masks (bool): Boolean for whether the model masks are being retrieved or not.
        initial (bool): Boolean for whether to use the initial directory. Default is False.

    Returns:
        str: Model's filepath.
    """
    return os.path.join(get_model_directory(seed, pruning_step, masks, initial), 'model.keras')

def create_path(path: str):
    """
    Helper function to create a path and all its subdirectories.

    Args:
        path (str): String containing the target path.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def initial_dir(parent_directory: str):
    """
    The path where the weights at the beginning of training are stored.

    Args:
        parent_directory (str): Parent directory where the initial weights directory will be created.

    Returns:
        str: Path to the initial weights directory.
    """
    return os.path.join(parent_directory, 'initial')

def final_dir(parent_directory: str):
    """
    The path where the weights at the end of training are stored.

    Args:
        parent_directory (str): Parent directory where the final weights directory will be created.

    Returns:
        str: Path to the final weights directory.
    """
    return os.path.join(parent_directory, 'final')

def mask_dir(parent_directory: str):
    """
    The path where the pruning masks are stored.

    Args:
        parent_directory (str): Parent directory where the masks directory will be created.

    Returns:
        str: Path to the masks directory.
    """
    return os.path.join(parent_directory, 'masks')

def weights_dir(parent_directory: str):
    """
    The path where the weights are stored.

    Args:
        parent_directory (str): Parent directory where the weights directory will be created.

    Returns:
        str: Path to the weights directory.
    """
    return os.path.join(parent_directory, 'weights')

def log_dir(parent_directory: str, name: str):
    """
    The path where training/testing/validation logs are stored.

    Args:
        parent_directory (str): Parent directory where the log directory will be created.
        name (str): Name of the log file.

    Returns:
        str: Path to the log file.
    """
    return os.path.join(parent_directory, '{}.log'.format(name))

def summaries_dir(parent_directory: str):
    """
    The path where TensorFlow summaries are stored.

    Args:
        parent_directory (str): Parent directory where the summaries directory will be created.

    Returns:
        str: Path to the summaries directory.
    """
    return os.path.join(parent_directory, 'summaries')

def trial_dir(parent_directory: str, trial_name: int):
    """
    The parent directory for a trial.

    Args:
        parent_directory (str): Directory the trial will go in.
        trial_name (int): Trial number to use as an index.

    Returns:
        str: Directory.
    """
    return os.path.join(parent_directory, f'{C.TRIAL_PREFIX}{trial_name}')
