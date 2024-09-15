"""
scripts/python/base.py

Base functionality used to generate plots
from external script.

Author: Jordan Bourdeau
"""

import functools
from importlib import reload
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os
from typing import Generator

from src.harness import constants as C
from src.harness import history
from src.harness import load
from src.metrics import experiment_aggregations as e_agg
from src.metrics import trial_aggregations as t_agg

from src.plotting import base_plots as bp
from src.plotting import global_plots as gp

def make_plots(
    batch_directory: str,
    target_directory: str,
    max_num_batches: int,
):
    # Hardcoded to use only the first one for now
    summaries = list(load.get_batch_summaries(batch_directory, max_num_batches=max_num_batches))[0]
    path = os.path.join(C.PLOTS_DIRECTORY, 'early_stopping_iteration.png')
    gp.plot_early_stopping(summary, save_location=path)
    print(summaries) 
    
    
