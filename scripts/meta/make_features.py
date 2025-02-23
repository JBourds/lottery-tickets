import os
import sys

epath = "/users/j/b/jbourde2/lottery-tickets/experiments/11-04-2024/lenet_mnist_0_seed_5_experiments_1_batches_0.025_default_sparsity_lm_pruning_20241102-111614"
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
output_path = os.path.join(root, "mnist_weightabase.pkl")

sys.path.append(root)
from src.metrics.features import *

if __name__ == "__main__":
    tdf, ldf, wdf = build_dataframes(epath)
    corrected_wdf = correct_class_imbalance(wdf)
    merged_df = merge_dfs(tdf, ldf, corrected_wdf)
    merged_df.to_pickle(output_path) 
