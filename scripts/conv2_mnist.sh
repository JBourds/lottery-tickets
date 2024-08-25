#!/usr/bin/env bash
cd scripts/
export PYTHONPATH=$PYTHONPATH:../
python3 iterative_pruning.py --model=conv2 --dataset=mnist "$@"

