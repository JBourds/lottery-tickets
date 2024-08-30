#!/usr/bin/env bash
cd scripts/training

if [[ $* == *--vacc* ]]; then
    echo 'Queueing job to VACC'
    export PYTHONPATH=$PYTHONPATH:../
    slaunch dggpu --rundir=../../experiments iterative_pruning.py --model=lenet --dataset=cifar "$@"
else
    echo 'Queueing job locally'
    export PYTHONPATH=$PYTHONPATH:../../
    python3 iterative_pruning.py --model=lenet --dataset=cifar "$@"
fi