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
pruning.py

Module containing function to prune masks based on magnitude.

Modified By: Jordan Bourdeau
Date: 3/17/24
"""

import numpy as np


def prune_by_percent(percents: dict[str: float], masks: dict[str: np.array], final_weights: dict[str: np.array]) -> dict[str: np.array]:
  """
  Return new masks that involve pruning the smallest of the final weights.

  :param percents:      A dictionary determining the percent by which to prune each layer.
                        Keys are layer names and values are floats between 0 and 1 (inclusive).
  :param masks:         A dictionary containing the current masks. Keys are strings and
                        values are numpy arrays with values in {0, 1}.
  :param final_weights: The weights at the end of the last training run. A
                        dictionary whose keys are strings and whose values are numpy arrays.

  :returns : A dictionary containing the newly-pruned masks.
  """

  def prune_by_percent_once(percent: float, mask: np.array, final_weight: np.array) -> np.array:
    # Put the weights that aren't masked out in sorted order.
    sorted_weights: np.array = np.sort(np.abs(final_weight[mask == 1]))

    # Determine the cutoff for weights to be pruned.
    cutoff_index: int = np.round(percent * sorted_weights.size).astype(int)
    cutoff: float = sorted_weights[cutoff_index]

    # Prune all weights below the cutoff.
    return np.where(np.abs(final_weight) <= cutoff, np.zeros(mask.shape), mask)

  new_masks = {}
  for k, percent in percents.items():
    new_masks[k] = prune_by_percent_once(percent, masks[k], final_weights[k])

  return new_masks
