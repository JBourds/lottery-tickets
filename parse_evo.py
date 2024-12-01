import functools
from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np
import os
import pickle
import re
from typing import Any, Dict
from tensorflow import keras

from src.harness import architecture as arch
from src.harness import constants as C
from src.harness import evolution as evo
from src.harness import utils
from scipy.stats import t  

EXPERIMENTS_DIRECTORY = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "experiments",
)

def parse_params(directory: str) -> Dict[str, Any]:
    # Parse parameters from each directory name
    pattern = r"evo_(.*?)_(.*?)_([0-9]+)_experiments_(None|[0-9]+(?:,[0-9]+)*)-(None|[a-zA-Z]+(?:,[a-zA-Z]+)*)_layers_([0-9]+)-asize_([0-9]+)-psize_([0-9]+)-nfronts_([0-9]+)-tsize_([0-9]+)-nwinners_((?:[0-9]*.)[0-9]+)-mscale_((?:[0-9]*.)[0-9]+)-mrate"
    match = re.match(pattern, directory)
    arch, dataset, num_experiments, layer_sizes, layer_activations, asize, psize, nfronts, tsize, nwinners, mscale, mrate = match.groups()
    if layer_sizes.lower() != "none":
        layer_sizes = list(map(int, layer_sizes.split(","))) 
    else:
        layer_sizes = [] 
    if layer_activations.lower() != "none":
        layer_activations = list(layer_activations.split(",")) 
    else:
        layer_activations = [] 
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
    evo_experiments = [
        directory for directory in os.listdir(EXPERIMENTS_DIRECTORY) 
        if directory.lower().startswith("evo") 
        and "5_experiments" in directory
        and "153114" in directory
    ]
    filenames = ["all_objective_metrics.pkl", "all_genome_metrics.pkl", "best_individuals.pkl"]
    experiment_metrics = {}
    best_index = 0
    # Keep track of all the data from trials
    mean_accuracies = []
    std_accuracies = []
    mean_sparsities = []
    std_sparsities = []
    max_accuracies = []
    max_sparsities = []
    exp_params = []

    best_accuracy = -float("inf")
    best_genome = None
    best_condition_index = None
    
    for condition_index, experiment_dir in enumerate(evo_experiments):
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
        max_accuracies.append(best_accuracies)
        max_sparsities.append(best_sparsities)
        
        highest_accuracy_experiment_index = np.argmax(best_accuracies)
        if best_accuracies[highest_accuracy_experiment_index] > best_accuracy:
            best_genome = best_genomes[highest_accuracy_experiment_index]
            best_accuracy = best_accuracies[highest_accuracy_experiment_index]
            best_condition_index = condition_index
    
    # Create plot showing results of gridsearch
    target_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        C.PLOTS_DIRECTORY,
        "gridsearch_results.png",
    )
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    def label_from_params(params: Dict) -> str:
        keys = ("layer_sizes", "layer_activations", "mscale", "mrate")
        layer_sizes, layer_activations, mscale, mrate = [params.get(key) for key in keys]
        if len(layer_sizes) == 0 and len(layer_activations) == 0:
            layer_string = "None"
        else:
            layer_string = list(zip(layer_sizes, layer_activations))
        label = f"Hidden Layers: {layer_string}\nScale: {mscale}\nRate: {mrate}"
        return label

    fig, (ci_plot, acc_boxplots, sparsity_boxplots) = plt.subplots(nrows=3, figsize=(8, 12))
    labels = list(map(label_from_params, exp_params))

    # Accuracy Boxplots
    acc_boxplots.set_title("Best Accuracy Distributions")
    acc_boxplots.set_ylabel("Accuracy (%)")
    acc_boxplots.yaxis.set_major_formatter(ticker.PercentFormatter())
    acc_boxplots.boxplot(max_accuracies, tick_labels=list(range(len(labels))))
    acc_boxplots.grid()

    # Sparsity Boxplots
    sparsity_boxplots.set_title("Sparsity Distributions at Best Accuracies")
    sparsity_boxplots.set_ylabel("Sparsity (%)")
    sparsity_boxplots.yaxis.set_major_formatter(ticker.PercentFormatter())
    sparsity_boxplots.boxplot(max_sparsities, tick_labels=list(range(len(labels))))
    sparsity_boxplots.grid()
    
    # CI Comparing Distributions 
    ci_plot.set_title("Grid Search Best Accuracy Comparisons")
    ci_plot.set_xlabel("Mean Sparsity (%)")
    ci_plot.set_ylabel("Mean Accuracy (%)")
    ci_plot.xaxis.set_major_formatter(ticker.PercentFormatter())
    ci_plot.yaxis.set_major_formatter(ticker.PercentFormatter())
    ci_plot.grid()

    for label_index, (label, mean_acc, std_acc, mean_spars, std_spars) in enumerate(zip(labels, mean_accuracies, std_accuracies, mean_sparsities, std_sparsities)):
        label = f"({label_index})\n{label}"
        n = len(mean_accuracies) 
        alpha = 0.05 
        t_value = t.ppf(1 - alpha / 2, df=n - 1) 

        # Create 95% CI for accuracy and sparsity at each point
        ci_plot.plot(mean_spars, mean_acc, label=label, marker=".", markersize=8)
        color = plt.gca().get_lines()[-1].get_color()
        lower_bound = mean_acc - t_value * (std_acc / np.sqrt(n))
        upper_bound = mean_acc + t_value * (std_acc / np.sqrt(n))
        ci_plot.fill_between([mean_spars], [lower_bound], [upper_bound], color=color, alpha=0.3)
        # Sparsity CI
        lower_bound = mean_spars - t_value * (std_spars / np.sqrt(n))
        upper_bound = mean_spars + t_value * (std_spars / np.sqrt(n))
        ci_plot.fill_betweenx([mean_acc], [lower_bound], [upper_bound], color=color, alpha=0.3)
    
    fig.tight_layout()
    fig.legend(loc="upper left", bbox_to_anchor=(1.05, 0.75))
    fig.savefig(target_path, bbox_inches="tight")
    
    # Take the very best genome and try it on different seeds for weight initializations
    print(f"Best condition index: {best_condition_index}\n{labels[best_condition_index]}")
    a = arch.Architecture("lenet", "mnist")
    params = exp_params[best_condition_index]
    layers = list(zip(params["layer_sizes"], params["layer_activations"]))
    feature_selectors = [
        evo.ModelFeatures.layer_sparsity,
        evo.ModelFeatures.magnitude,
        evo.ModelFeatures.random,
        functools.partial(evo.ModelFeatures.synaptic_flow, loss_fn=keras.losses.CategoricalCrossentropy()),
    ]
    
    arch_feature_selectors = [
        evo.ArchFeatures.layer_num,
        evo.ArchFeatures.layer_ohe,
        evo.ArchFeatures.layer_prop_params,
    ] 
    individual_constructor = functools.partial(
        evo.Individual,
        "lenet",
        "mnist",
        model_feature_selectors=feature_selectors,
        arch_feature_selectors=arch_feature_selectors,
        layers=layers,
    )
    reinit_accuracies = []
    reinit_sparsities = []
    for seed in range(10, 30):
        utils.set_seed(seed) 
        individual = individual_constructor()
        # Change this out on later trials so it is not hardcoded
        # individual.genome = best_genome 
        evo.Individual.update_phenotype(individual)
        reinit_accuracies.append(evo.Individual.eval_accuracy(individual))
        reinit_sparsities.append(evo.Individual.sparsity(individual))
    
    label = labels[best_condition_index] 
    target_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        C.PLOTS_DIRECTORY,
        "transfer_masking.png",
    )
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    fig, (acc_boxplot, sparsity_boxplot) = plt.subplots(ncols=2)
    fig.suptitle("Transfer Masking With Best Genome")

    acc_boxplot.set_title("Accuracy")
    bplot1 = acc_boxplot.boxplot(reinit_accuracies)["boxes"][0]
    acc_boxplot.yaxis.set_major_formatter(ticker.PercentFormatter())
    acc_boxplot.set_ylabel("Accuracy %")
    acc_boxplot.set_xlabel("")
    acc_boxplot.axhline(0.10)
    acc_boxplot.grid()

    sparsity_boxplot.set_title("Sparsity")
    bplot2 = sparsity_boxplot.boxplot(reinit_sparsities)["boxes"][0]
    sparsity_boxplot.yaxis.set_major_formatter(ticker.PercentFormatter())
    sparsity_boxplot.set_ylabel("Sparsity (%)")
    sparsity_boxplot.set_xlabel("")
    sparsity_boxplot.axhline(0.50)
    sparsity_boxplot.grid()

    handles = [bplot1, bplot2]
    fig.legend(handles, [label], loc="upper center", bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    fig.savefig(target_path, bbox_inches="tight")
