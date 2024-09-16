import os
import pickle

from src.harness import history

p = 'experiments/lenet/models/model_0/trial0/trial_data.pkl'

path = 'experiments/lenet/'
trial_data = history.get_experiments(path)
for experiment in trial_data:
    for trial in experiment:
        for key, t in zip(trial.__dict__.keys(), map(type, trial.__dict__.values())):
            print(key, t)
            if t == list:
                print(type(trial.__dict__[key][0]))
        

