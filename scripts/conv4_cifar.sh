#!/usr/bin/env bash
cd scripts/training

if [[ $* == *--vacc* ]]; then
    echo 'Queueing job to VACC'
    export PYTHONPATH=$PYTHONPATH:../
    slaunch dggpu --rundir=../../experiments iterative_pruning.py --model=conv4 --dataset=cifar "$@"
else
    echo 'Queueing job locally'
    export PYTHONPATH=$PYTHONPATH:../../
    python3 iterative_pruning.py --model=conv4 --dataset=cifar "$@"
fi