#!/usr/bin/env bash
SCRIPT_PATH=$(dirname $(dirname $(dirname $(realpath ${0}))))
ROOT_PATH=$(dirname $SCRIPT_PATH)
cd ${SCRIPT_PATH}/plotting/python
EXPERIMENT_DIRECTORY=$(python3 get_dir.py "$@")

if [[ $* == *--vacc* ]]; then
    echo 'Creating plots on the VACC'
    export PYTHONPATH=$PYTHONPATH:$ROOT_PATH
    slaunch blue --rundir=$EXPERIMENT_DIRECTORY make_plots.py "$@"
else
    echo 'Creating plots locally'
    export PYTHONPATH=$PYTHONPATH:$ROOT_PATH
    python3 make_plots.py "$@"
fi
