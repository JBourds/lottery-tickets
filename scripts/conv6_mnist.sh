#!/usr/bin/env bash
cd scripts/training

if [[ $* == *--vacc* ]]; then
    echo 'Queueing job to VACC'
    export PYTHONPATH=$PYTHONPATH:../
    slaunch dggpu --rundir=../../experiments iterative_pruning.py --model=conv6 --dataset=mnist "$@"
else
    echo 'Queueing job locally'
    export PYTHONPATH=$PYTHONPATH:../../
    python3 iterative_pruning.py --model=conv6 --dataset=mnist "$@"
fi