import os
import pandas as pd
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import sys
from tqdm import tqdm
from typing import List

basepath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(basepath)

from src.harness.meta import *
from src.metrics.features import *

def make_plot(accuracies: npt.NDArray[np.float32], columns: List[str], batch_size: int = 256, location: str = "single_feature_analysis.png"):
    indices = np.argsort(np.max(accuracies, axis=1))[::-1]
    plt.figure(figsize=(10, 8))
    plt.title(f"Meta Mask Single Feature Importance (Batch Size = {batch_size})")
    plt.ylabel("Accuracy")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.xlabel("Epoch")
    for index in indices:
        accuracy = accuracies[index]
        feature = columns[index]
        plt.plot(np.arange(epochs), accuracy, label=f"{feature}: {np.max(accuracy) * 100:.2f}%")
    plt.legend()
    plt.savefig("single_feature_importances.png")

if __name__ == "__main__":
    df_path = os.path.join(basepath, "weightabase.pkl")
    merged_df = pd.read_pickle(df_path)

    i_features = ["l_sparsity", "l_rel_size", "li_prop_positive", "wi_std", "wi_perc", "wi_synflow", "wi_sign", "dense", "bias", "conv", "output"]
    epochs = 3
    batch_size = 256
    histories = []
    # for index, feature in tqdm(enumerate(merged_df.columns)):
    #     print(f"{feature} ({index + 1} / {len(merged_df.columns)})")
    #     iX, iY = featurize_db(merged_df, [feature])
    #     i_meta = create_meta(iX[0].shape)
    #     histories.append(i_meta.fit(iX, iY, epochs=epochs, batch_size=batch_size, validation_split=0.2, shuffle=True))

    # accuracies = np.array(list(map(lambda x: x.history["accuracy"], histories)))
    # with open("columns.txt", "w") as outfile:
    #     outfile.write(str(merged_df.columns))
    # np.save("accuracies", accuracies)

    accuracies = np.load("accuracies.npy")
    with open("columns.txt", "r") as infile:
        contents = infile.read()
        columns = eval(contents[contents.index("["):contents.index("]")+1])
    make_plot(accuracies, columns)

