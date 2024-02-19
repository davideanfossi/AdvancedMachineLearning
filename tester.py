import numpy as np
import pandas as pd
import sys
from utils.temporal_aggregation import aggregate_features
from utils.extract_pkl import *
from utils.dataset_creator import *

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
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')

    # FEATURE EXTRACTION (Step #2)
    # aggregate_features("train")
    # extract_pkl(pkl_folder)

    # DATA EXTRACTION (Step #3)
    """
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
            print("\n")"""
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
    #df, _, df_emg = rgb_action_net_creation("train_val_action_net/D4", "train_val/D4", "train_val_action_net/D4_emg")
    
   #df = get_data_from_pkl_pd("train_val_action_net/D4_train")
   #df_emg = get_data_from_pkl_pd("train_val_action_net/D4_emg_train")
   #print(df.head(), "\n")
   #print(df_emg.head(), "\n")
   #print(len(df_emg.right_readings[0]), "\n")
    
    #data = get_data_from_pkl_pd("train_val/D3_train")
    #print(set(data["description"]))
    #print(len(set(data["verb"])))
    #print(df[["video_id", "verb", "start_timestamp", "stop_timestamp", "start_frame", "stop_frame"]])
    #print(len(set(df["uid"])))

    folder = [
        "action-net/pickles/S00_2",
        "action-net/pickles/S01_1",
        "action-net/pickles/S02_2",
        "action-net/pickles/S02_3",
        "action-net/pickles/S02_4",
        "action-net/pickles/S03_1",
        "action-net/pickles/S03_2",
        "action-net/pickles/S05_2",
        "action-net/pickles/S06_1",
        "action-net/pickles/S06_2",
        "action-net/pickles/S07_1",
        "action-net/pickles/S08_1",
        "action-net/pickles/S09_2",
    ]
    """df = emg_analysis(folder)
    max_r =  max([len(x) for x in df["right_readings"]])
    min_r = min([len(x) for x in df["right_readings"]])
    max_l = max([len(x) for x in df["left_readings"]])
    min_l = min([len(x) for x in df["left_readings"]])
    mean_l = statistics.median([len(x) for x in df["left_readings"]])
    mean_r = statistics.median([len(x) for x in df["right_readings"]])
    #count_min_l = sum([len(x) < mean_l - 200 for x in df["right_readings"]])
    #count_min_r = sum([len(x) < mean_l - 200 for x in df["left_readings"]])
    #count_max_l = sum([len(x) > mean_l + 200 for x in df["right_readings"]])
    #count_max_r = sum([len(x) > mean_l + 200 for x in df["left_readings"]])
    print("LEN tot right ", (len(df["right_readings"])))
    print("LEN tot left ", (len(df["left_readings"])))
    print("LEN MAX right: ", max_r)
    print("LEN MIN right: ", min_r)
    print("LEN MAX left: ", max_l)
    print("LEN MIN left: ", min_l)
    print("MEAN LEN right: ", mean_r)
    print("MEAN LEN left: ", mean_l, "\n")"""
    #print("COUNT MIN right: ", count_min_r)
    #print("COUNT MIN left: ", count_min_l)
    #print("COUNT MAX right: ", count_max_r)
    #print("COUNT MAX left: ", count_max_l)

    # emg_dataset("action-net/ActionNet_train", "train_val/big_file_train.pkl")
    # emg_dataset("action-net/ActionNet_test", "train_val/big_file_test.pkl")
    # data = get_data_from_pkl_pd("train_val/big_file_train")
    # data = pd.DataFrame(data["features"])
    # print(data.head(), "\n")
    # data = get_data_from_pkl_pd("train_val/big_file_test")
    # data = pd.DataFrame(data["features"])
    # print(data.head())


    # BIG FILE WITH PREPROCESSED DATA AND SPRECTROGRAM
    emg_dataset_spettrogram("action-net/ActionNet_train", "train_val/big_file_train_spe.pkl")
    emg_dataset_spettrogram("action-net/ActionNet_test", "train_val/big_file_test_spe.pkl")
    data = get_data_from_pkl_pd("train_val/big_file_train_spe")
    data = pd.DataFrame(data["features"])
    print(data.head(), "\n")
    data = get_data_from_pkl_pd("train_val/big_file_test_spe")
    data = pd.DataFrame(data["features"])
    print(data.head())


    # HDF5 handler (Step #3)
    # hdf5_handler()
