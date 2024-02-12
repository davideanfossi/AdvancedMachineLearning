import numpy as np
import pandas as pd
import sys
from utils.temporal_aggregation import aggregate_features
from utils.extract_pkl import *
from utils.extract_hdf5 import *
from utils.rgb_dataset import *

pkl_folder = [
    "saved_features/aggregated_test_D5",
    "saved_features/aggregated_test_D10",
    "saved_features/aggregated_test_D25",
    "saved_features/aggregated_test_U5",
    "saved_features/aggregated_test_U10",
    "saved_features/aggregated_test_U25",
]

pkl_folder_action_net = ["action-net/ActionNet_test", "action-net/ActionNet_train"]

hdf5_filepath = "../data/action_net/EMG/ActionNet Wearables S04.hdf5"
hdf5_out_folder = "../data/action_net/EMG/view"


def hdf5_handler():
    keys = [
        "in_path",
        "out_folder",
        "mode",
        "device",
        "stream",
    ]

    params = {
        "in_path": hdf5_filepath,
        "out_folder": hdf5_out_folder,  # --> offset +2
    }

    # ? Passing parameters with this setup:
    # python tester.py <in_path> <out_folder> <mode> <device> <stream>

    # ? offset used to avoid passing all parameters
    offset = len(params)

    if len(sys.argv) > 1:
        for i in range(1, len(sys.argv)):
            params[keys[i - 1 + offset]] = sys.argv[i]

    hdf5_extractor = HDF5Extractor(*params.values())
    hdf5_extractor.extract_hdf5(save=False, label_index=1)


if __name__ == "__main__":
    # FEATURE EXTRACTION (Step #2)
    # aggregate_features("train")
    # extract_pkl(pkl_folder)

    # DATA EXTRACTION (Step #3)
    # data = get_data_from_pkl_pd("train_val_an/S04_1")
    # all_columns = data.columns.tolist()
    # row = data.iloc[-1]
    # value = row['start']
    # print(row)
    # print(value)

    # RGB Action Net Creation
    rgb_action_net_creation("train_val_an/D1_train")
    data = get_data_from_pkl_pd("train_val_an/D1_train")
    print(data)

    # HDF5 handler (Step #3)
    # hdf5_handler()
