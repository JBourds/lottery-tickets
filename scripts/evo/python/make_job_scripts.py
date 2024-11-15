
import os

TRAINING_DIRECTORY = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
OUTPUT_DIRECTORY = os.path.join(TRAINING_DIRECTORY, 'shell')
RUN_DIRECTORY = os.path.dirname(os.path.dirname(TRAINING_DIRECTORY))

models = ['lenet']
datasets = ['mnist'] 

for model in models:
    for dataset in datasets:
        file = os.path.join(OUTPUT_DIRECTORY, f'{model}_{dataset}.sh')
        with open(file, 'w') as script:
            contents = f"""#!/usr/bin/env bash
cd {os.path.join(TRAINING_DIRECTORY, 'python')}

if [[ $* == *--vacc* ]]; then
    echo 'Queueing job to VACC'
    export PYTHONPATH=$PYTHONPATH:{RUN_DIRECTORY}
    rundir=$(python3 make_dir.py --aname={model} --dname={dataset} "$@")
    mkdir -p $rundir
    echo $rundir
    slaunch dggpu --rundir=$rundir main.py --aname={model} --dname={dataset} --rundir=$rundir "$@"
else
    echo 'Queueing job locally'
    export PYTHONPATH=$PYTHONPATH:{RUN_DIRECTORY}
    rundir=$(python3 make_dir.py --aname={model} --dname={dataset} "$@")
    mkdir -p $rundir
    python3 main.py --aname={model} --dname={dataset} --rundir=$rundir "$@"
fi"""

            script.write(contents)   
        os.chmod(file, 0o775)

filepath = os.path.join(OUTPUT_DIRECTORY, 'run_all.sh')
with open(filepath, 'w') as outfile:
    contents = ""
    for model in models:
        for dataset in datasets:
            contents += os.path.join(OUTPUT_DIRECTORY, model + '_' + dataset + '.sh') + ' "$@" &\n'
    outfile.write(contents)
os.chmod(filepath, 0o775)
