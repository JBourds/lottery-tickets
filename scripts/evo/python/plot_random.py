import numpy as np
from matplotlib import pyplot as plt

random_accuracies = np.load("random_accuracies.npy").flatten()
random_sparsities = np.load("random_sparsities.npy").flatten()
print(f"Random Accuracies:\n{random_accuracies}")
print(f"Random Sparsities:\n{random_sparsities}")

best_accuracy_index = np.argmax(random_accuracies)
print(f"Highest Accuracy index: {best_accuracy_index}")
print(f"Best Accuracy: {random_accuracies[best_accuracy_index]}, Sparsity: {random_sparsities[best_accuracy_index]}")

best_sparsity_index = np.argmin(random_sparsities)
print(f"Lowest sparsity index: {best_sparsity_index}")
print(f"Best Sparsity: {random_sparsities[best_sparsity_index]}, Accuracy: {random_accuracies[best_sparsity_index]}")

