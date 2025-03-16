
import itertools
import os

TRAINING_DIRECTORY = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__)))
OUTPUT_DIRECTORY = os.path.join(TRAINING_DIRECTORY, 'shell')
RUN_DIRECTORY = os.path.dirname(os.path.dirname(TRAINING_DIRECTORY))
models = ['lenet', 'conv2', 'conv4', 'conv6']
datasets = ['mnist', 'fashion_mnist', 'cifar']
# initializers = ["glorot_uniform", "glorot_normal",
#                 "random_normal", "random_uniform", "he_normal", "he_uniform"]
initializers = ["glorot_uniform", "glorot_normal"]
rewind_rules = ["oi", "sc", "no"]
pruning_rules = ["lm", "hm"]
sparsity_strategies = ["default", "slow", "fast"]


for model in models:
    for dataset in datasets:
        file = os.path.join(OUTPUT_DIRECTORY, f'{model}_{dataset}.sh')
        with open(file, 'w') as script:
            contents = f"""#!/usr/bin/env bash
cd {os.path.join(TRAINING_DIRECTORY, 'python')}

if [[ $* == *--vacc* ]]; then
    echo 'Queueing job to VACC'
    export PYTHONPATH=$PYTHONPATH:{RUN_DIRECTORY}
    rundir=$(python3 make_dir.py --model={model} --dataset={dataset} "$@")
    mkdir -p $rundir
    echo $rundir
    slaunch nvgpu --rundir=$rundir iterative_pruning.py --model={model} --dataset={dataset} --rundir=$rundir "$@"
else
    echo 'Queueing job locally'
    export PYTHONPATH=$PYTHONPATH:{RUN_DIRECTORY}
    rundir=$(python3 make_dir.py --model={model} --dataset={dataset} "$@")
    mkdir -p $rundir
    python3 iterative_pruning.py --model={model} --dataset={dataset} --rundir=$rundir "$@"
fi"""

            script.write(contents)
        os.chmod(file, 0o775)

filepath = os.path.join(OUTPUT_DIRECTORY, 'run_all.sh')
with open(filepath, 'w') as outfile:
    contents = ""
    for model, dataset, initializer, rewind_rule, pruning_rule, sparsity_strategy in itertools.product(
            models, datasets, initializers, rewind_rules, pruning_rules, sparsity_strategies):
        cmd = os.path.join(OUTPUT_DIRECTORY,
                           model + '_' + dataset + '.sh')
        cmd += f" --initializer={initializer}"
        cmd += f" --rewind_rule={rewind_rule}"
        cmd += f" --pruning_rule={pruning_rule}"
        cmd += f" --sparsity_strategy={sparsity_strategy}"
        cmd += " \"$@\" &"
        contents += f"{cmd}\n"
    outfile.write(contents)
os.chmod(filepath, 0o775)
