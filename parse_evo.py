from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np
import os
import pickle
from typing import Any, Dict
import re

from src.harness import constants as C
from src.harness import evolution as evo
from scipy.stats import t  # Import t-distribution for CI calculation

EXPERIMENTS_DIRECTORY = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "experiments",
)

def parse_params(directory: str) -> Dict[str, Any]:
    # Parse parameters from each directory name
    pattern = r"evo_(.*?)_(.*?)_([0-9]+)_experiments_(None|[0-9]+(?:,[0-9]+)*)-(None|[a-zA-Z]+(?:,[a-zA-Z]+)*)_layers_([0-9]+)-asize_([0-9]+)-psize_([0-9]+)-nfronts_([0-9]+)-tsize_([0-9]+)-nwinners_((?:[0-9]*.)[0-9]+)-mscale_((?:[0-9]*.)[0-9]+)-mrate"
    match = re.match(pattern, directory)
    arch, dataset, num_experiments, layer_sizes, layer_activations, asize, psize, nfronts, tsize, nwinners, mscale, mrate = match.groups()
    params = {
        "arch": arch,
        "dataset": dataset,
        "num_experiments": num_experiments,
        "layer_sizes": layer_sizes,
        "layer_activations": layer_activations,
        "asize": asize,
        "psize": psize,
        "nfronts": nfronts,
        "tsize": tsize,
        "nwinners": nwinners,
        "mscale": mscale,
        "mrate": mrate,
    }
    return params

if __name__ == "__main__":
    evo_experiments = [directory for directory in os.listdir(EXPERIMENTS_DIRECTORY) if directory.lower().startswith("evo") and "5_experiments" in directory]
    filenames = ["all_objective_metrics.pkl", "all_genome_metrics.pkl", "best_individuals.pkl"]
    experiment_metrics = {}
    best_index = 0
    mean_accuracies = []
    std_accuracies = []
    mean_sparsities = []
    std_sparsities = []
    exp_params = []
    for experiment_dir in evo_experiments:
        results = []
        params = parse_params(experiment_dir)
        exp_params.append(params)
        for filename in filenames:
            try:
                with open(os.path.join(EXPERIMENTS_DIRECTORY, experiment_dir, filename), "rb") as infile:
                    data = pickle.load(infile)
                    results.append(data)
            except Exception as e:
                print(f"Error {e} Skipping:", filename, "for directory:", experiment_dir)
        obj_metrics, genome_metrics, best_genomes = results
        experiment_metrics[experiment_dir] = results
        best_accuracy_indexes = list(map(lambda obj_results: np.argmax(obj_results["objective_0_value"]), obj_metrics)) 
        best_accuracies = [obj_results["objective_0_value"].flatten()[index] for obj_results, index in zip(obj_metrics, best_accuracy_indexes)]
        best_sparsities = [obj_results["objective_1_value"].flatten()[index] for obj_results, index in zip(obj_metrics, best_accuracy_indexes)]
        mean_accuracies.append(np.mean(best_accuracies))
        mean_sparsities.append(np.mean(best_sparsities))
        std_accuracies.append(np.std(best_accuracies))
        std_sparsities.append(np.std(best_sparsities))
    
    # Create plot showing results of gridsearch
    target_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        C.PLOTS_DIRECTORY,
        "gridsearch_results.png",
    )
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    plt.title("Grid Search Parameter Results")
    
    for params, mean_acc, std_acc, mean_spars, std_spars in zip(exp_params, mean_accuracies, std_accuracies, mean_sparsities, std_sparsities):
        keys = ("layer_sizes", "layer_activations", "mscale", "mrate")
        layer_sizes, layer_activations, mscale, mrate = [params.get(key) for key in keys]
        if layer_sizes.lower() == "none" and layer_activations.lower() == "none":
            layer_string = "None"
        else:
            layer_string = list(zip(layer_sizes.split(","), layer_activations.split(",")))
        label = f"Hidden Layers: {layer_string}\nScale: {mscale}\nRate: {mrate}"
       
        n = len(mean_accuracies) 
        alpha = 0.05 
        t_value = t.ppf(1 - alpha / 2, df=n - 1) 

        # Create 95% CI for accuracy and sparsity at each point
        plt.plot(mean_spars, mean_acc, label=label, marker=".", markersize=8)
        color = plt.gca().get_lines()[-1].get_color()
        lower_bound = mean_acc - t_value * (std_acc / np.sqrt(n))
        upper_bound = mean_acc + t_value * (std_acc / np.sqrt(n))
        plt.fill_between([mean_spars], [lower_bound], [upper_bound], color=color, alpha=0.3)
        # Sparsity CI
        lower_bound = mean_spars - t_value * (std_spars / np.sqrt(n))
        upper_bound = mean_spars + t_value * (std_spars / np.sqrt(n))
        plt.fill_betweenx([mean_acc], [lower_bound], [upper_bound], color=color, alpha=0.3)
    
    plt.xlabel("Mean Sparsity (%)")
    plt.ylabel("Mean Accuracy (%)")
    plt.tight_layout()
    plt.gca().xaxis.set_major_formatter(ticker.PercentFormatter())
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter())
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.savefig(target_path, bbox_inches="tight")
