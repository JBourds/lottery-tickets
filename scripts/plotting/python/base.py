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
    summary = load.get_batch_summaries(batch_directory, max_num_batches=max_num_batches)
    print("Hello world!")
    x = np.arange(10)
    y = np.arange(10)
    plt.plot(x, y)
    plt.savefig('test')
