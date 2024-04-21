"""
paths.py

Utilities for building paths to store data.

Modified By: Jordan Bourdeau
Date: 3/17/24
"""

import os

def get_model_directory(seed: int, pruning_step: int, masks: bool = False) -> str:
    """
    Function used to retrieve a model's path.

    :param seed:           Random seed the model was trained using
    :param pruning_step:   Integer value for the number of pruning steps which had been completed for the model.
    :param masks:          Boolean for whether the model masks are being retrieved or not.

    :returns: Model directory.
    """
    output_directory: str = get_model_directory(seed, C.MODEL_DIRECTORY)
    trial_directory: str = trial(output_directory, pruning_step)
    target_directory: str = masks(trial_directory) if masks else weights(trial_directory)
    return target_directory

def get_model_filepath(seed: int, pruning_step: int, masks: bool = False) -> str:
    """
    Function used to retrieve a model's file path

    :param seed:         Random seed the model was trained using
    :param pruning_step: Integer value for the number of pruning steps which had been completed for the model.
    :param masks:        Boolean for whether the model masks are being retrieved or not.

    :returns: Model's filepath.
    """
    return os.path.join(get_model_directory(seed), f'step_{pruning_step}{"_masks" if masks else ""}', 'model.keras')

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

def get_model_directory(model_index: int, parent_directory: str = "models") -> str:
    """
    Function to return the relative directory where a model would go.

    :param parent_directory: Base directory to append model subdirectory to. Defaults to empty string.
    :param model_index:      Integer for the index/random seed of the model.

    :returns: Returns expected directory for the model.
    """
    return os.path.join(parent_directory, f'model_{model_index}')

def initial(parent_directory):
  """The path where the weights at the beginning of training are stored."""
  return os.path.join(parent_directory, 'initial')


def final(parent_directory):
  """The path where the weights at the end of training are stored."""
  return os.path.join(parent_directory, 'final')


def masks(parent_directory):
  """The path where the pruning masks are stored."""
  return os.path.join(parent_directory, 'masks')

def weights(parent_directory):
  """The path where the weights are stored."""
  return os.path.join(parent_directory, 'weights')


def log(parent_directory, name):
  """The path where training/testing/validation logs are stored."""
  return os.path.join(parent_directory, '{}.log'.format(name))


def summaries(parent_directory):
  """The path where tensorflow summaries are stored."""
  return os.path.join(parent_directory, 'summaries')


def trial(parent_directory, trial_name):
  """The parent directory for a trial."""
  return os.path.join(parent_directory, f'trial{trial_name}')


def run(parent_directory,
        level,
        experiment_name,
        run_id=''):
  """The name for a particular training run.

  Args:
    parent_directory: The directory in which this directory should be created.
    level: The number of pruning iterations.
    experiment_name: The name of this specific experiment.
    run_id: (optional) The number of this run (if the same experiment is being
      run more than once).

  Returns:
    The path in which data about this run should be stored.
  """
  return os.path.join(parent_directory, str(level),
                      f'{experiment_name}{run_id}')
