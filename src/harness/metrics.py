"""
metrics.py

Module for computing metrics on lottery ticket similarities.
"""

import tensorflow as tf

def get_train_test_loss_accuracy() -> tuple[tf.keras.metrics.Metric, ...]:
    """
    Create metrics to measure the error and accuracy of the model.
    """
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    return train_loss, train_accuracy, test_loss, test_accuracy