#!/usr/bin/env bash
SCRIPT_PATH=$(dirname $(dirname $(realpath ${0})))
EXPERIMENT_PATH=$(dirname $SCRIPT_PATH)
cd ${SCRIPT_PATH}/python

if [[ $* == *--vacc* ]]; then
    echo 'Creating plots on the VACC'
    export PYTHONPATH=$PYTHONPATH:../../../
    slaunch blue --rundir=$EXPERIMENT_PATH make_plots.py "$@"
else
    echo 'Creating plots locally'
    export PYTHONPATH=$PYTHONPATH:../../../
    python3 make_plots.py "$@"
fi
