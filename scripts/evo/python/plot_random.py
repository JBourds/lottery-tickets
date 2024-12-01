from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np
import os
import sys

sys.path.append(os.path.join(
    os.path.expanduser("~"),
    "lottery-tickets",
))

from src.harness import constants as C

generations = 1000
steps = 10

random_accuracies = np.load(f"random_accuracies_{generations}_gens_{steps}_steps.npy").flatten()
random_sparsities = np.load(f"random_sparsities_{generations}_gens_{steps}_steps.npy").flatten()

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
fig.suptitle("Results from 1000 Random Masks")

accuracy_plot.boxplot(random_accuracies)
accuracy_plot.set_title("Accuracy")
accuracy_plot.set_xlabel("")
accuracy_plot.set_ylabel("Accuracy (%)")
accuracy_plot.yaxis.set_major_formatter(ticker.PercentFormatter())
accuracy_plot.axhline(0.1)

sparsity_plot.boxplot(random_sparsities)
sparsity_plot.set_title("Sparsity")
sparsity_plot.set_xlabel("")
sparsity_plot.set_ylabel("Sparsity (%)")
sparsity_plot.yaxis.set_major_formatter(ticker.PercentFormatter())
sparsity_plot.axhline(0.5)

caption = f"Highest Accuracy: {random_accuracies[best_accuracy_index]:.2%} at {random_sparsities[best_accuracy_index]:.2%} sparsity"
# fig.text(0, 0, caption, va="top")
fig.tight_layout()

target_path = os.path.join(
    os.path.expanduser("~"),
    "lottery-tickets",
    C.PLOTS_DIRECTORY,
    f"random_weights_{generations}_gens_{steps}_steps.png"
)
os.makedirs(os.path.dirname(target_path), exist_ok=True)

fig.savefig(target_path, bbox_inches="tight")
