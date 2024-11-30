import numpy as np
from matplotlib import pyplot as plt

random_accuracies = np.load("random_accuracies.npy").flatten()
random_sparsities = np.load("random_sparsities.npy").flatten()
# print(f"Random Accuracies:\n{random_accuracies}")
# print(f"Random Sparsities:\n{random_sparsities}")

best_accuracy_index = np.argmax(random_accuracies)
print(f"Highest Accuracy index: {best_accuracy_index}")
print(f"Best Accuracy: {random_accuracies[best_accuracy_index]}, Sparsity: {random_sparsities[best_accuracy_index]}")

best_sparsity_index = np.argmin(random_sparsities)
print(f"Lowest sparsity index: {best_sparsity_index}")
print(f"Best Sparsity: {random_sparsities[best_sparsity_index]}, Accuracy: {random_accuracies[best_sparsity_index]}")

mean_accuracy = np.mean(random_accuracies)
mean_sparsity = np.mean(random_sparsities)

print(f"Mean accuracy: {mean_accuracy}")
print(f"Mean sparsity: {mean_sparsity}")

fig, (accuracy_plot, sparsity_plot) = plt.subplots(ncols=2)
fig.suptitle("Results from 1000 Random Trials")

accuracy_plot.boxplot(random_accuracies)
accuracy_plot.set_title("Accuracy")
sparsity_plot.boxplot(random_sparsities)
sparsity_plot.set_title("Sparsity")

caption = f"Highest Accuracy: {random_accuracies[best_accuracy_index]:.2%} at {random_sparsities[best_accuracy_index]:.2%} sparsity"
# fig.text(0, 0, caption, va="top")
fig.tight_layout()
fig.savefig("random_plot.png", bbox_inches="tight")
