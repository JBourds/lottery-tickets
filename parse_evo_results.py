import os
import pickle

EXPERIMENTS_DIRECTORY = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "experiments",
)

evo_experiments = [directory for directory in os.listdir(EXPERIMENTS_DIRECTORY) if directory.lower().startswith("evo")]
filenames = ["all_objective_metrics.pkl", "all_genome_metrics.pkl", "all_archives.pkl"]
experiment_metrics = {}
for experiment_dir in evo_experiments:
    results = []
    for filename in filenames:
        try:
            with open(os.path.join(EXPERIMENTS_DIRECTORY, experiment_dir, filename), "rb") as infile:
                data = pickle.loads(infile)
                results.append(data)
        except Exception as e:
            print(f"Skipping:", filename, "for directory:", experiment_dir)
    experiment_metrics[experiment_dir] = results

print(experiment_metrics)

    
