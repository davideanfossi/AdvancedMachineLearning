import pandas as pd
import pickle
from utils.extract_pkl import get_data_from_pkl
import numpy
import math

def get_data_from_pkl_pd(pkl_file):
    # Open the .pkl file in binary mode for reading
    with open(f"{pkl_file}.pkl", "rb") as pkl_file:
        # Load the data from the .pkl file
        data = pd.read_pickle(pkl_file)

    return data

def save_into_big_pkl(action_net_path, big_file_path):
    actionNet_train = get_data_from_pkl_pd(action_net_path);
    data = {"features": []}
    len_time = 1000
    
    # read each row of actionNet_train
    for i in range(len(actionNet_train)):
        index = actionNet_train.index[i]
        file = actionNet_train.iloc[i].file
        label = actionNet_train.iloc[i].description

        id = file + "_" + str(index)

        # get readings and timestamps from the file
        Spkl = get_data_from_pkl_pd("readings/" + file.strip(".pkl"))
        right_readings = Spkl.myo_right_readings[index]
        left_readings = Spkl.myo_left_readings[index]
        right_timestamps = Spkl.myo_right_timestamps[index]
        left_timestamps = Spkl.myo_left_timestamps[index]
        
        # calculate len_time
        if len(left_readings) < len(right_readings):
            split = len(left_readings) / len_time
            len_time_left = len_time
            len_time_right = math.ceil(len(right_readings) / split )
        else:
            split = len(right_readings) / len_time
            len_time_right = len_time
            len_time_left = math.ceil(len(left_readings) / split)

        # separate the readings and timestamps into len_time_left and len_time_right intervals
        right_readings = [right_readings[i:i+len_time_right] for i in range(0, len(right_readings), len_time_right)]
        left_readings = [left_readings[i:i+len_time_left] for i in range(0, len(left_readings), len_time_left)]

        if split > 2:
            right_readings[-2] = numpy.concatenate((right_readings[-2], right_readings[-1]), axis=0)
            right_readings.pop(-1)

            left_readings[-2] = numpy.concatenate((left_readings[-2], left_readings[-1]), axis=0)
            left_readings.pop(-1)   

        #[print(len(r), len(l)) for r, l in zip(right_readings, left_readings)]

        for i in range(len(right_readings)): 
            data["features"].append({
                "id": id + "_" + str(i),
                "right_readings": right_readings[i],
                "left_readings": left_readings[i],
                "label": label,
            })

        #print(i + 1, "of", len(actionNet_train), "done")

    
    big_file = open(big_file_path, "wb")
    pickle.dump(data, big_file)
    big_file.close()


def main():
    save_into_big_pkl("action-net/ActionNet_train", "train_val/big_file_train.pkl")
    save_into_big_pkl("action-net/ActionNet_test", "train_val/big_file_test.pkl")

    #test_train = get_data_from_pkl("big_file_train")
    #print(len(test_train["features"]))

    #test_test = get_data_from_pkl("big_file_test")
    #print(len(test_test["features"]))

if __name__ == '__main__':
    main()
