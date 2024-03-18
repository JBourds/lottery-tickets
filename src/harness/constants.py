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
constants.py

File containing all constants and constant functions.
"""

import functools
import tensorflow as tf

# File Locations
DATA_DIRECTORY: str = 'data/'
MODEL_DIRECTORY: str = 'models/'
CHECKPOINT_DIRECTORY: str = 'checkpoints/'
FIT_DIRECTORY: str = 'logs/fit/'

DIRECTORIES: list[str] = [
    MODEL_DIRECTORY,
    CHECKPOINT_DIRECTORY,
    FIT_DIRECTORY,
    DATA_DIRECTORY,
]

# Experiment Parameters
NUM_MODELS: int = 1

# Training Parameters
OPTIMIZER = functools.partial(tf.keras.optimizers.legacy.SGD, .1)
MNIST_LOCATION: str = DATA_DIRECTORY + 'mnist/'

# This is small for now in order to perform testing, original paper used 60,000 iterations
TRAINING_ITERATIONS: int = 1_000
PRUNING_STEPS: int = 45
# Prune each layer by 10%
PRUNING_PERCENTS: dict[int: float] = {f'layer{i}': 0.1 for i in range(3)}

TEST_PRUNING_STEPS: int = 5
TEST_TRAINING_ITERATIONS: int = 10