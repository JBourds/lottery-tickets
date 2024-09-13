for f in ./scripts/*.sh;
    do bash "$f" --target_sparsity=0.05 --experiments=20 --batches=2 --vacc --max_processes=1;
done
