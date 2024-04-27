"""
paths.py

Utilities for building paths to store data.

Modified By: Jordan Bourdeau
Date: 3/17/24
"""

import os

import src.harness.constants as C

def get_model_directory(seed: int, pruning_step: int, masks: bool = False, initial: bool = False, parent_directory: str = C.MODEL_DIRECTORY) -> str:
    """
    Function used to retrieve a model's path.

    :param seed:             Random seed the model was trained using
    :param pruning_step:     Integer value for the number of pruning steps which had been completed for the model.
    :param masks:            Boolean for whether the model masks are being retrieved or not. Default is False
    :param initial:          Boolean for whether to use the initial directory. Default is False.
    :param parent_directory: String for the parent directory of the model.
  
    :returns: Model directory.
    """
    output_directory: str = os.path.join(parent_directory, f'model_{seed}')
    trial_directory: str = initial_dir(output_directory) if initial else trial_dir(output_directory, pruning_step)
    target_directory: str = mask_dir(trial_directory) if masks else weights_dir(trial_directory)
    return target_directory

def get_model_filepath(seed: int, pruning_step: int, masks: bool = False, initial: bool = False) -> str:
    """
    Function used to retrieve a model's file path

    :param seed:         Random seed the model was trained using
    :param pruning_step: Integer value for the number of pruning steps which had been completed for the model.
    :param masks:        Boolean for whether the model masks are being retrieved or not.
    :param initial:      Boolean for whether to use the initial directory. Default is False.

    :returns: Model's filepath.
    """
    return os.path.join(get_model_directory(seed, pruning_step, masks, initial), 'model.keras')

def create_path(path: str):
    """
    Helper function to create a path and all its subdirectories.
    :param path: String containing the target path.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created successfully.")
    else:
        print(f"Directory '{path}' already exists.")

def initial_dir(parent_directory):
  """The path where the weights at the beginning of training are stored."""
  return os.path.join(parent_directory, 'initial')

def final_dir(parent_directory):
  """The path where the weights at the end of training are stored."""
  return os.path.join(parent_directory, 'final')

def mask_dir(parent_directory):
  """The path where the pruning masks are stored."""
  return os.path.join(parent_directory, 'masks')

def weights_dir(parent_directory):
  """The path where the weights are stored."""
  return os.path.join(parent_directory, 'weights')

def log_dir(parent_directory, name):
  """The path where training/testing/validation logs are stored."""
  return os.path.join(parent_directory, '{}.log'.format(name))

def summaries_dir(parent_directory):
  """The path where tensorflow summaries are stored."""
  return os.path.join(parent_directory, 'summaries')

def trial_dir(parent_directory, trial_name):
  """The parent directory for a trial."""
  return os.path.join(parent_directory, f'trial{trial_name}')
