"""
scripts/make_plots.py

Command line script to generate all the plots for a given set of models
into a target directory.

Author: Jordan Bourdeau
"""

import argparse
import datetime
import functools
import logging
import os
import sys
from typing import List

import numpy as np

from scripts.plotting.python.base import make_plots
from src.harness import constants as C

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script to generate all model plots.')
    # Experiment params
    parser.add_argument("--root", type=str, default=None,
        help="Root directory for the experiments to create plots for.")
    parser.add_argument("--models_dir", type=str, default=C.MODELS_DIRECTORY,
        help="Directory storing models.")
    parser.add_argument("--plots_dir", type=str, default=C.PLOTS_DIRECTORY,
        help="Output directory to store plots in.")
    parser.add_argument("--eprefix", type=str, default=C.EXPERIMENT_PREFIX,
        help="Prefix for experiment directories (single random seeds).")
    parser.add_argument("--tprefix", type=str, default=C.TRIAL_PREFIX,
        help="Prefix for trial directories (rounds of IMP).")
    parser.add_argument("--tdata", type=str, default=C.TRIAL_DATAFILE,
        help="Name for the datafile in trial directories.")
    parser.add_argument("--seeding_rule", type=str, default="",
                        help="Rule for how weights were seeded.")
 
    args, unknown = parser.parse_known_args() 
    if args.root is None:
        raise ValueError("Must provide root directory for experiments.")
    make_plots(
        args.root, 
        args.models_dir, 
        args.plots_dir, 
        args.eprefix, 
        args.tprefix, 
        args.tdata,
        args.seeding_rule,
    )

