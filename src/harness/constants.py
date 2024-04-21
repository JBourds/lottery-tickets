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

class Constants:
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

    # Training Parameters
    MNIST_LOCATION: str = DATA_DIRECTORY + 'mnist/'

    PATIENCE: int = 3
    MINIMUM_DELTA: float = 0.01
    LEARNING_RATE: float = 0.1
    OPTIMIZER = functools.partial(tf.keras.optimizers.legacy.Adam, LEARNING_RATE)

    # Test Experiment Parameters
    TEST_NUM_MODELS: int = 2
    TEST_TRAINING_EPOCHS: int = 2
    TEST_PRUNING_STEPS: int = 2

    # Real Experiment Parameters
    NUM_MODELS: int = 100
    TRAINING_EPOCHS: int = 30
    PRUNING_STEPS: int = 45
