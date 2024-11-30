"""
training/seeding_gridsearch.py

Script for running a gridsearch over different seeding
parameters. Runs VACC jobs in the background.

Author: Jordan Bourdeau
"""

from itertools import product
import numpy as np
import os
import subprocess

# Grid search params
models = ["lenet"]
datasets = ["mnist"]
targets = ["hm"]
# percentages = np.array([1, 5, 10, 25])
percentages = np.array(["00005", "0001", "0005", "001", "001"])
scalars = [1.5, 2.5, 5]
constants = [-1, 1]

# Sign-aware constants
signs = ["s", "p", "fp"]

# General params
experiment_params = "--target_sparsity=0.025 --experiments=5 --vacc"

# # Test with single case
# percentages = np.array(["00005"])
# scalars = [1.5]
# constants = []
# signs = ["p"]
# experiment_params = "--target_sparsity=0.65 --experiments=1 --vacc"

print(f"Common params: {experiment_params}")
print("Grid search over the following parameters:")
print(f"Models: {models}")
print(f"Datasets: {datasets}")
print(f"Targets: {targets}")
print(f"Percentages: {percentages}")
print(f"Scalars: {scalars}")
print(f"Constants: {constants}")

def run_all(experiment_params: str, seeding_rule: str):
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        "shell",
        "run_all.sh"
    )
    cmd = f"nohup {script_path} {seeding_rule} {experiment_params}" \
         + " {0} >/dev/null 2>&1 &"
    print(cmd)
    subprocess.call(cmd, shell=True)
    
    
def run_one(model: str, dataset: str, experiment_params: str, seeding_rule: str):
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        "shell",
        f"{model}_{dataset}.sh"
    )
    cmd = f"nohup {script_path} {seeding_rule} {experiment_params}" \
        + " {0} >/dev/null 2>&1 &"
    print(cmd)
    subprocess.call(cmd, shell=True)
    

# Scaling
for model, dataset, target, percent, sign, scalar in product(models, datasets, targets, percentages, signs, scalars):
    seeding_rule = f"--seeding_rule={target}{percent},{sign},scale{scalar}"
    run_one(model, dataset, experiment_params, seeding_rule)

# Constants
for model, dataset, target, percent, sign, constant in product(models, datasets, targets, percentages, signs, constants):
    seeding_rule = f"--seeding_rule={target}{percent},{sign}set{constant}"
    run_one(model, dataset, experiment_params, seeding_rule)
