"""
constants.py

File containing all constants and constant functions.
"""

from enum import Enum
import functools
import os
# from tensorflow.keras.losses import CategoricalCrossentropy
# from tensorflow.keras.optimizers import Adam
from sys import platform

# Directories
DATA_DIRECTORY = 'data'
MODELS_DIRECTORY = 'models'
EXPERIMENT_PREFIX = 'model'
TRIAL_PREFIX = 'trial'
TRIAL_DATAFILE = 'trial_data.pkl'
EXPERIMENTS_DIRECTORY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'experiments',
)
PLOTS_DIRECTORY = 'plots'

# PATIENCE = 2
# MINIMUM_DELTA = 0.0001
# LEARNING_RATE = 0.005
# OPTIMIZER = functools.partial(Adam, LEARNING_RATE)
# 
# LOSS_FUNCTION = functools.partial(CategoricalCrossentropy)
# 
# # Test Experiment Parameters
# TEST_NUM_MODELS = 2
# TEST_TRAINING_EPOCHS = 2
# TEST_PRUNING_STEPS = 2
# 
# # Real Experiment Parameters
# NUM_MODELS = 100
# TRAINING_EPOCHS = 60
# BATCH_SIZE = 128
# PRUNING_STEPS = 45
# 
# # Architectures
# MLP_ARCHITECTURES = {
#    'lenet',
# }
# 
# SUPPORTED_ARCHITECTURES = {
#    'lenet',
#    'conv-2',
#    'conv-4',
#    'conv-6',
# }
# 
# class OriginalParams(Enum):
#    """
#    Class acting as a namespace for original LTH paper parameters
#    by Frankle and Carbin.
#    
#    Taken largely from figure 2.
#    """
#    
#    LEARNING_RATE = 0.0012
#    OPTIMIZER = functools.partial(Adam, LEARNING_RATE)
#        
#    LENET_BATCH_SIZE = 60
#    RESNET_BATCH_SIZE = 128
#    VGG_BATCH_SIZE = 64
#    
#    CONV_PRUNING_RATE = 0.1
#    FC_PRUNING_RATE = 0.2
#    # The number of iterations (batches) to train for at a time before evaluating
#    # validation set performance and checking for early stopping
#    PERFORMANCE_EVALUATION_FREQUENCY = 100
