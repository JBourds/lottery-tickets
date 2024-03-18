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
training.py

Module containing function to train a neural network.

Modified By: Jordan Bourdeau
Date: 3/17/24
"""

import numpy as np
import tensorflow as tf

from src.harness.model import save_model
from src.lottery_ticket.foundations.model_base import ModelBase

def train(make_dataset: callable, model: ModelBase, optimizer: tf.keras.optimizers.Optimizer, epochs: int):
    """
    Function to perform training for a model.

    :param make_dataset:     Function to produce the training/test sets.
    :param model:            Model to optimize.
    :param optimizer:        Optimizer to use during training.
    :param epochs:           Number of epochs to train for.

    :returns: A dictionary of the weights before and after training.
    """
    # Update list of variables to optimize
    initial_weights: dict[str: np.array] = model.get_current_weights()
    # optimize = optimizer.minimize(model.loss, var_list=initial_weights)

    # Get handles?

    # Save summaries

    # Save network initial weights and masks
    save_model(model, 0, False)
