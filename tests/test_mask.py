import numpy as np
import os
import sys
import tensorflow as tf

sys.path.append("../")

from src.harness import architecture as arch
from src.harness import history
from src.harness import utils

for i in range(3):
    root = f"/users/j/b/jbourde2/lottery-tickets/experiments/lenet_mnist_0_seed_3_experiments_1_batches_0.05_default_sparsity_lm_pruning_20241015-053957/models/model{i}"
    first = history.TrialData.load_from(os.path.join(root, "trial6", "trial_data.pkl"))
    last = history.TrialData.load_from(os.path.join(root, "trial6", "trial_data.pkl"))

    a = arch.Architecture("lenet", "mnist")
    X_train, X_test, Y_train, Y_test = a.load_data()
    utils.set_seed(i)
    model = a.get_model_constructor()()
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )
    initial_weights = model.get_weights()

    # Verify initial weights are right
    assert np.all([np.array_equal(f, l) for f, l in zip(first.initial_weights, last.initial_weights)])
    assert np.all([np.array_equal(f, l) for f, l in zip(first.initial_weights, initial_weights)])

    # Test masked weight performance
    print("Initial weights validation accuracy")
    print(f"Model sparsity: {utils.model_sparsity(model) * 100:.2f}%")
    model.evaluate(X_test, Y_test)
    model.set_weights([w * m for w, m in zip(initial_weights, last.masks)])
    print(f"Model sparsity: {utils.model_sparsity(model) * 100:.2f}%")
    print("Masked initial weights validation accuracy")
    model.evaluate(X_test, Y_test)

    mw = [mw[mw != 0] for mw in model.get_weights()]
    iw = [iw[(iw * mask) != 0] for iw, mask in zip(last.initial_weights, last.masks)]
    assert np.all([np.array_equal(w1, w2) for w1, w2 in zip(mw, iw)])
