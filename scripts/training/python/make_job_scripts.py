
import os

TRAINING_DIRECTORY = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIRECTORY = os.path.join(TRAINING_DIRECTORY, 'shell')
RUN_DIRECTORY = os.path.dirname(os.path.dirname(TRAINING_DIRECTORY))

models = ['lenet', 'conv2', 'conv4', 'conv6']
datasets = ['mnist', 'cifar'] 

for model in models:
    for dataset in datasets:
        file = os.path.join(OUTPUT_DIRECTORY, f'{model}_{dataset}.sh')
        with open(file, 'w') as script:
            contents = f"""#!/usr/bin/env bash
cd {os.path.join(TRAINING_DIRECTORY, 'python')}

if [[ $* == *--vacc* ]]; then
    echo 'Queueing job to VACC'
    export PYTHONPATH=$PYTHONPATH:{RUN_DIRECTORY}
    rundir=$(python3 make_dir.py "$0")
    mkdir $rundir
    echo $rundir
    slaunch dggpu --rundir=$rundir iterative_pruning.py --model={model} --dataset={dataset} --rundir=$rundir "$@"
else
    echo 'Queueing job locally'
    export PYTHONPATH=$PYTHONPATH:{RUN_DIRECTORY}
    rundir=$(python3 make_dir.py "$0")
    mkdir $rundir
    python3 iterative_pruning.py --model={model} --dataset={dataset} --rundir=$rundir "$@"
fi"""

            script.write(contents)   
        os.chmod(file, 0o775)

