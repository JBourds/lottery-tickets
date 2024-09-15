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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script to generate all model plots.')
    # Experiment params
    parser.add_argument('--dir', type=str, default=None,
                        help='Output directory to store plots in.')

    args, unknown = parser.parse_known_args() 
    make_plots(args.dir, ".", 1)
