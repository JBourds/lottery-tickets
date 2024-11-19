#!/usr/bin/env bash

FILE_DIR=$(dirname $(realpath ${0}))
SCRIPT_PATH=$(dirname $(dirname $(dirname $(realpath ${0}))))
ROOT_PATH=$(dirname $SCRIPT_PATH)

# Takes one argument: root of directory with all the experiments
EXPERIMENT_ROOT=$1

# Loop through each directory in the specified root path
for dir in $(ls -1 "$EXPERIMENT_ROOT"); do
    full_dir="$EXPERIMENT_ROOT/$dir"
echo $full_dir
    
    if [ -d "$full_dir" ]; then
        # Check for the presence of a "models" directory 
        # Rename files starting with "slurm*" to "training_output*"
        # so it does not complain about there already being a slurm file
        for file in "$full_dir"/slurm*; do
            if [ -e "$file" ]; then
                base_name="${file##*/}"
                new_name="training_output${base_name#slurm}"
                mv "$file" "$full_dir/$new_name"
                echo "Renamed $file to $new_name"
            fi
        done

        # Change to the directory and call make_plots.sh
        echo "Calling make_plots.sh for $full_dir"
        cd "$full_dir" || exit 1  # Exit if unable to change directory
        $FILE_DIR/make_plots.sh --root=$full_dir --dir=$full_dir --vacc
        cd - || exit 1  # Return to the previous directory
    fi
done

