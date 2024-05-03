"""
load.py

Module containing functions for working with trial data to
load experiments and experiment summaries.

Author: Jordan Bourdeau
Date Created: 5/2/24
"""

import os
import re

from src.harness import history

def get_experiment_from_trials(
    experiment_directory: str,
    trial_directory_prefix: str = 'trial',
    trial_data_file: str = 'trial_data.pkl',
    ) -> history.ExperimentSummary:
    """
    Function which takes an experiment directory and the expected prefix for
    each trial directory within it and reconstructs an `ExperimentData` object.
    
    Expects the following convention:
        - Within the directory for each experiment, each trial (round of pruning) follows the
          naming convention <prefix><step> counting up from 0.

    Args:
        experiment_directory (str): Directory the experiment is in.
        trial_directory_prefix (str, optional): Prefix for the trial directory. Defaults to 'trial'.
        trial_data_file (str, optional): Name of the trial data file. Defaults to 'trial_data.pkl'.

    Returns:
        history.ExperimentSummary: Object containing a list of all pruning rounds.
    """
    
    # Get a list of all files in the experiment directory which case insensitively match trial prefix
    trial_directories: list[str] = [directory for directory in os.listdir(experiment_directory) if re.match(trial_directory_prefix, directory, re.IGNORECASE)]
    
    # For each directory, try to read the trial data with `history.TrialData.load_from(filepath)` into a list
    experiment_data: history.ExperimentData = history.ExperimentData()
    trials: list[history.TrialData] = [history.TrialData.load_from(os.path.join(experiment_directory, trial_directory, trial_data_file)) for trial_directory in trial_directories]
    experiment_data.add_trials(trials)
    
    return experiment_data

def get_batch_summaries(
    batch_directory: str, 
    batch_prefix: str = 'batch_',
    batch_data_file: str = 'experiment_summary.pkl',
    start_batch: int = 0,
    max_num_batches: int = None,
    ) -> list[history.ExperimentSummary]:
    """
    Function which takes a top level directory and information about the naming convention of 
    subdirectories in order to read in all the batch experiment summaries.
    
    Expects the following convention:
        - Within the directory for each experiment, the naming convention is <prefix><model index/seed>
          for each model.
        - Within the directory for each experiment, each trial (round of pruning) follows the
          naming convention <prefix><step> counting up from 0.

    Args:
        batch_directory (str): Directory where batches were stored.
        batch_prefix (str, optional): Prefix for each batch. Defaults to 'batch_'.
        batch_data_file (str, optional): Name of batch experiment summary. Defaults to 'experiment_summary.pkl'.
        start_batch: (int): Starting index of the batch to run. Defaults to 0.
        max_num_batches (int, optional): Maximum number of batch summaries to try and load.
            Default is None, will load every possible batch.

    Returns:
        list[history.ExperimentSummary]: List of all experiment summaries which were read in.
    """
        
    if not isinstance(start_batch, int):
        raise ValueError('Start batch must be an integer index for target batch.')
    elif start_batch < 0:
        raise ValueError('Start batch cannot be negative.')
    elif max_num_batches is None:
        max_num_batches = float('inf')
    
        
    # Get a list of all directories in the experiment directory which match the batch prefix
    batch_summaries: list[history.ExperimentSummary] = []
    # Order batch directories by number at the end (lexicographical ordering doesn't work here)
    batch_directories: list[str] = sorted([directory for directory in os.listdir(batch_directory) if re.match(batch_prefix, directory, re.IGNORECASE)], key=lambda x: int(re.match(batch_prefix + r'(\d+)$', x, re.IGNORECASE).group(1)))
    
    for batch_dir in batch_directories[start_batch:]:
        batch_number = int(re.match(batch_prefix + r'(\d+)$', batch_dir, re.IGNORECASE).group(1))
        if batch_number >= start_batch and batch_number < start_batch + max_num_batches:
            batch_summary_file: str = os.path.join(batch_directory, batch_dir, batch_data_file)
            if not os.path.exists(batch_summary_file):
                raise FileNotFoundError(f'Could not find summary file at path: {batch_summary_file}')
            batch_summaries.append(history.ExperimentSummary.load_from(batch_summary_file))
    
    return batch_summaries

def get_experiment_summaries_from_batch_trials(
    batch_directory: str, 
    batch_prefix: str = 'batch_',
    experiments_directory: str = 'models',
    experiment_prefix: str = 'model_',
    trial_prefix: str = 'trial',
    trial_data_file: str = 'trial_data.pkl',
    max_num_experiments: int = None,
    ) -> list[history.ExperimentSummary]:
    """
    Function which takes a top level directory and information about the naming convention of 
    subdirectories in order to create a list of experiment summaries corresponding to the 
    number of batches.
    
    Expects the following convention:
        - Within the directory for each experiment, the naming convention is <prefix><model index/seed>
          for each model.
        - Within the directory for each experiment, each trial (round of pruning) follows the
          naming convention <prefix><step> counting up from 0.

    Args:
        batch_directory (str): Directory where batches were stored.
        batch_prefix (str, optional): Prefix for each batch. Defaults to 'batch_'.
        experiments_directory (str, optional): Directory experiments are in. Defaults to 'models'.
        experiment_prefix (str, optional): Prefix for the directory an experiment is in. Defaults to 'model_'.
        trial_prefix (str, optional): Prefix for the directory a trial is in. Defaults to 'trial'.
        trial_data_file (str, optional): Name of the trial data file. Defaults to 'trial_data.pkl'.
        max_num_experiments (int, optional): Maximum number of experiments to try and load.
            Default is None, will load every possible experiment.

    Returns:
        list[history.ExperimentSummary]: List of all experiment summaries which were read in.
    """
    if max_num_experiments is None:
        max_num_experiments = float('inf')
        
    # Get a list of all files in the experiment directory which case insensitively match trial prefix
    batch_summaries: list[history.ExperimentSummary] = []
    batch_directories: list[str] = [directory for directory in os.listdir(batch_directory) if re.match(batch_prefix, directory, re.IGNORECASE)]
    experiment_count: int = 0
    for batch_dir in batch_directories:
        if experiment_count >= max_num_experiments:
            break
        
        experiment_directory: str = os.path.join(batch_directory, batch_dir, experiments_directory)
        batch_summary: history.ExperimentSummary = load_summary_from_trials(
            experiment_directory=experiment_directory,
            experiment_prefix=experiment_prefix,
            trial_prefix=trial_prefix,
            trial_data_file=trial_data_file,
            max_num_trials=max_num_experiments - experiment_count
        )
        
        experiment_count += batch_summary.get_experiment_count()
        
        batch_summaries.append(batch_summary)
    
    return batch_summaries

def load_summary_from_trials(
    experiment_directory: str,
    experiment_prefix: str = 'model_',
    trial_prefix: str = 'trial',
    trial_data_file: str = 'trial_data.pkl',
    max_num_trials: int = None,
) -> history.ExperimentSummary:
    """
    Helper function which loads an ExperimentSummary object when pointed to a directory
    which contains trials in it.

    Args:
        experiment_directory (str): Directory containing experiments.
        experiment_prefix (str, optional): Prefix to each individual experiment directory name. Defaults to 'model_'.
        trial_prefix (str, optional): Prefix for trial directories. Defaults to 'trial'.
        trial_data_file (str, optional): Expected name for trial data files. Defaults to 'trial_data.pkl'.
        max_num_trials (int, optional): Maximum number of trials to include in summary. Defaults to None.

    Returns:
        history.ExperimentSummary: ExperimentSummary object created from the trials read in.
    """
    
    if max_num_trials is None:
        max_num_trials = float('inf')
    
    summary: history.ExperimentSummary = history.ExperimentSummary()
    experiment_directories: list[str] = [
        directory for directory in 
        os.listdir(experiment_directory) 
        if re.match(experiment_prefix, directory, re.IGNORECASE)
    ]
    
    trial_count: int = 0
    for experiment_dir in experiment_directories:
        if trial_count >= max_num_trials:
            break
    
        model_seed: int = int(experiment_dir.split(experiment_prefix, 1)[1])
        experiment_data: history.ExperimentData = get_experiment_from_trials(
            experiment_directory=os.path.join(experiment_directory, experiment_dir),
            trial_directory_prefix=trial_prefix,
            trial_data_file=trial_data_file
        )
        
        summary.add_experiment(model_seed, experiment_data)
        trial_count += 1
        
    return summary