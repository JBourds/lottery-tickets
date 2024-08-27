if ! conda env list | grep -q 'lottery-tickets'; then
    conda env create -f env.yml -n lottery-tickets
fi
