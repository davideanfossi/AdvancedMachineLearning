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
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')

    # FEATURE EXTRACTION (Step #2)
    # aggregate_features("train")
    # extract_pkl(pkl_folder)

    # DATA EXTRACTION (Step #3)
    data = get_data_from_pkl_pd("action-net/S04_1")
    for i in range(1, len(data)):
        data_l = int(data["myo_left_timestamps"][i][0])
        data_r = int(data["myo_right_timestamps"][i][0])
        data_l_end = int(data["myo_left_timestamps"][i][-1])
        data_r_end = int(data["myo_right_timestamps"][i][-1])
        data_start = int(data["start"][i])
        data_stop = int(data["stop"][i])

        diff_l = data_l_end - data_l
        diff_r = data_r_end - data_r
        diff_start = data_start - data_l
        if diff_l != diff_r or diff_start != 0:
            print("left start ", data_l)
            print("right start ",data_r)
            print("left end ", data_l_end)
            print("right end ", data_r_end)
            print("data start ", data_start)
            print("data stop ", data_stop)
            print("diff left ", diff_l)
            print("diff right ", diff_r)
            print("diff start ", diff_start)
            print("\n")
    #print(data[["description", "start", "stop"]])
    #data = get_data_from_pkl_pd("action-net/ActionNet_train")
    #all_columns = data.columns.tolist()
    #print(all_columns)
    #row = data['description']
    # value = row['start']
    #print(set(row))
    #for i in range(60):
        #if len(data['myo_right_timestamps'][i]) != len(data['myo_left_timestamps'][i]):
            #print(len(data['myo_right_timestamps'][i])-len(data['myo_left_timestamps'][i]))
    #print(data[['myo_right_timestamps', 'myo_left_timestamps']])

    # RGB Action Net Creation
    df, _, df_emg = rgb_action_net_creation("train_val_action_net/D4", "train_val_action_net/D4_reduced", "train_val_action_net/D4_emg")
    #data = get_data_from_pkl_pd("train_val/D3_train")
    #print(set(data["description"]))
    #print(len(set(data["verb"])))
    #print(df[["video_id", "verb", "start_timestamp", "stop_timestamp", "start_frame", "stop_frame"]])
    #print(len(set(df["uid"])))

    # HDF5 handler (Step #3)
    # hdf5_handler()
