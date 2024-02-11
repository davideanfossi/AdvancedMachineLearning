import numpy as np
import sys
from utils.temporal_aggregation import aggregate_features
from utils.extract_pkl import *
from utils.extract_hdf5 import *

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

if __name__ == "__main__":
    # FEATURE EXTRACTION
    # aggregate_features("train")
    # extract_pkl(pkl_folder)

    # HDF5 handler
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
    # python tester.py <save> <in_path> <out_folder> <mode> <device> <stream>

    # ? offset used to avoid passing all parameters
    offset = 1 + len(params)

    if len(sys.argv) > 1:
        for i in range(offset, len(sys.argv)):
            params[keys[i - offset + 1]] = sys.argv[i]

    hdf5_extractor = HDF5Extractor(*params.values())
    hdf5_extractor.extract_hdf5(save=False)
