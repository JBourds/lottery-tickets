
import os

OUTPUT_DIRECTORY = os.path.dirname(os.path.dirname(__file__))

models = ['lenet', 'conv2', 'conv4', 'conv6']
datasets = ['mnist', 'cifar'] 

for model in models:
    for dataset in datasets:
        file = os.path.join(OUTPUT_DIRECTORY, f'{model}_{dataset}.sh')
        with open(file, 'w') as script:
            contents = f"""#!/usr/bin/env bash
cd scripts/training

if [[ $* == *--vacc* ]]; then
    echo 'Queueing job to VACC'
    export PYTHONPATH=$PYTHONPATH:../
    slaunch dggpu --rundir=../../experiments iterative_pruning.py --model={model} --dataset={dataset} "$@"
else
    echo 'Queueing job locally'
    export PYTHONPATH=$PYTHONPATH:../../
    python3 iterative_pruning.py --model={model} --dataset={dataset} "$@"
fi"""
            script.write(contents)
       
        os.chmod(file, 0o775)

