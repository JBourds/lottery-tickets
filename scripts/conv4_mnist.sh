#!/usr/bin/env bash
cd scripts/
export PYTHONPATH=$PYTHONPATH:../
python3 iterative_pruning.py --model=conv4 --dataset=mnist "$@"

