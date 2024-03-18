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

import tensorflow as tf

def train(make_dataset: callable, create_model: callable, optimizer: tf.keras.optimizers.Optimizer, epochs: int):
    """
    Function to perform training for a model.

    :param make_dataset: Function to produce the training/test sets.
    :param create_model: Function to produce the model.
    :param optimizer:    Optimizer to use during training.
    :param epochs:       Number of epochs to train for.

    :returns: A dictionary of the weights before and after training.
    """
    pass