import pandas as pd
import pickle
from utils.extract_pkl import get_data_from_pkl
import numpy
import math
from pprint import pprint

def get_data_from_pkl_pd(pkl_file):
    # Open the .pkl file in binary mode for reading
    with open(f"{pkl_file}.pkl", "rb") as pkl_file:
        # Load the data from the .pkl file
        data = pd.read_pickle(pkl_file)

    return data

def save_into_big_pkl(action_net_path, big_file_path, label_dict):
    actionNet_train = get_data_from_pkl_pd(action_net_path);
    data = {"features": []}
    len_time = 100
    
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
  
        # separate the readings into bloks of 100 and truncate longer readings 
        minimum = min(len(right_readings), len(left_readings))
        right_readings = [right_readings[i:i+len_time] for i in range(0, minimum, len_time)]
        left_readings = [left_readings[i:i+len_time] for i in range(0, minimum, len_time)]

        # pop the last reading if it is smaller than len_time
        if len(right_readings[-1]) < len_time or len(left_readings[-1]) < len_time:
            right_readings.pop(-1)
            left_readings.pop(-1)  

        #[print(len(r), len(l)) for r, l in zip(right_readings, left_readings)]
            
        # check shape is correct
        for i in range(len(right_readings)):
            assert right_readings[i].shape == (len_time, 8)
            assert left_readings[i].shape == (len_time, 8)

        for i in range(len(right_readings)): 
            data["features"].append({
                "id": id + "_" + str(i),
                "right_readings": right_readings[i],
                "left_readings": left_readings[i],
                "label": label_dict[label],
            })

        #print(i + 1, "of", len(actionNet_train), "done")

    big_file = open(big_file_path, "wb")
    pickle.dump(data, big_file)
    big_file.close()


def main():
    label_dict = {
        "Spread jelly on a bread slice": 0,
        "Slice a potato": 1,
        'Get items from refrigerator/cabinets/drawers': 2,
        "Get/replace items from refrigerator/cabinets/drawers": 2,
        "Clean a plate with a towel": 3,
        "Pour water from a pitcher into a glass": 4,
        "Stack on table: 3 each large/small plates, bowls": 5,
        "Spread almond butter on a bread slice": 6,
        "Slice a cucumber": 7,
        "Clean a pan with a sponge": 8,
        "Load dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils": 9,
        'Open a jar of almond butter': 10,
        "Open/close a jar of almond butter": 10,
        "Slice bread": 11,
        "Peel a cucumber": 12,
        "Clean a plate with a sponge": 13,
        "Clear cutting board": 14,
        "Set table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils": 15,
        "Clean a pan with a towel": 16,
        "Peel a potato": 17,
        "Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils": 18,
        "Get items from cabinets: 3 each large/small plates, bowls, mugs, glasses, sets of utensils": 19
    }
    
    save_into_big_pkl("action-net/ActionNet_train", "train_val/big_file_train.pkl",label_dict)
    save_into_big_pkl("action-net/ActionNet_test", "train_val/big_file_test.pkl", label_dict)

    # test_train = get_data_from_pkl("train_val/big_file_train")
    # dict = {}
    # for sample in test_train["features"]:
    #     if sample["label"] in dict:
    #         dict[sample["label"]] += 1
    #     else:
    #         dict[sample["label"]] = 1

    # print(dict)

    #print(test_train["features"][0])

    #test_train = get_data_from_pkl("train_val/big_file_train")
    # for sample in test_train["features"]:
    #     print(sample["label"])
    #print(test_train["features"][:]["label"])

    # test_test = get_data_from_pkl("train_val/big_file_test")
    # print(len(test_test["features"]))


if __name__ == '__main__':
    main()