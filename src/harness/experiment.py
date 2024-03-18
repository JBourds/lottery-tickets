# Copyright (C) 2018 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
experiment.py

Module containing code for actually running the lottery ticket hypothesis experiemnts.

Modified By: Jordan Bourdeau
Date: 3/17/24
"""

import numpy as np

from src.harness.model import LeNet300, save_model

def experiment(make_dataset: callable, 
               make_model: callable, 
               train_model: callable, 
               prune_masks: callable, 
               pruning_steps: int,
               presets: dict[str: np.array] = None):
  """
  Run the lottery ticket experiment for the specified number of iterations.

  :param make_dataset:  A function that, when called with no arguments, will create the training and test sets.
  :param make_model:    A function that, when called with four arguments (input_tensor,
                        label_tensor, presets, masks), creates a model object that descends from
                        model_base. Presets and masks are optional.
  :param train_model:   A function that, when called with three arguments (pruning iteration number, 
                        tuple with dataset training/test sets, model), trains the model using the
                        dataset and returns the model's initial and final weights as dictionaries.
  :param prune_masks:   A function that, when called with two arguments (dictionary of
                        current masks, dictionary of final weights), returns a new dictionary of
                        masks that have been pruned. Each dictionary key is the name of a tensor
                        in the network; each value is a numpy array containing the values of the
                        tensor (1/0 values for mask, weights for the dictionary of final weights).
  :param pruning_steps: The number of pruning iterations to perform.
  :param presets:       (optional) The presets to use for the first iteration of training.
                        In the form of a dictionary where each key is the name of a tensor and
                        each value is a numpy array of the values to which that tensor should
                        be initialized.
  """

  # A helper function that trains the network once according to the behavior
  # determined internally by the+ train_model function.
  def train_once(pruning_step: int, presets=None, masks=None):
    print(f'Pruning Step {pruning_step}')
    X_train, Y_train, _, _ = make_dataset()
    model: LeNet300 = make_model(X_train, Y_train, presets=presets, masks=masks)
    return train_model(make_dataset, model, pruning_step)

  # Run once normally.
  initial_weights, final_weights = train_once(0, presets=presets)

  # Create the initial masks with no weights pruned.
  masks = {}
  for k, v in initial_weights.items():
    masks[k] = np.ones(v.shape)

  # Begin the training loop.
  for pruning_step in range(1, pruning_steps + 1):
    # Prune the network.
    masks = prune_masks(masks, final_weights)
    # Train the network again.
    _, final_weights = train_once(pruning_step, presets=initial_weights, masks=masks)
