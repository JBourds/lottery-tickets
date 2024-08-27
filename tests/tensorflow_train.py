"""
Use pure tensorflow training to compare against results from custom training
loop.
"""

import sys

import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow import keras

sys.path.append('../')


if __name__ == '__main__':
    from src.harness.architecture import Architecture, Hyperparameters
    architecture = Architecture('conv2', 'cifar')
    model = architecture.get_model_constructor()()
    dataset = architecture.dataset
    hyperparams = architecture.get_model_hyperparameters()
    X_train, X_test, Y_train, Y_test = dataset.load()

    model.compile(
        optimizer=hyperparams.optimizer(hyperparams.learning_rate),
        loss=hyperparams.loss_function(),
        metrics=[
            hyperparams.accuracy_metric()
        ],
    )

    model.fit(X_train, Y_train, epochs=hyperparams.epochs, batch_size=hyperparams.batch_size,
              validation_steps=hyperparams.eval_freq, validation_split=0.1, callbacks=[
                  tf.keras.callbacks.EarlyStopping(
                      min_delta=hyperparams.minimum_delta, patience=hyperparams.patience)
              ])

    model.summary()

    results = model.evaluate(X_test, Y_test, batch_size=hyperparams.batch_size)
