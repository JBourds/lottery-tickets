import os
import pickle

from src.harness import history
from src.metrics import trial_aggregations as t_agg

path = 'experiments/lenet_mnist_10'
trial_data = history.get_experiments(path)
experiment_aggs = []
for experiment in trial_data:
    for trial in experiment:
        print(trial.final_weights[-1]) 
        

