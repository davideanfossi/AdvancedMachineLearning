import random
import pickle
import math
import pandas as pd
import numpy as np
from utils.extract_pkl import *
from sklearn.model_selection import train_test_split


"""labels_reduced = {
    "Spread": "Spread",
    "Open": "Open/Close",
    "Clean": "Clean",
    "Unload": "Unload",
    "Clear": "Clear",
    "Set": "Set",
    "Load": "Load",
    "Stack": "Stack",
    "Get": "Get/Put",
    "Pour": "Pour",
    "Peel": "Peel",
    "Slice": "Slice",
}
record["verb"] = labels[(row["description"].split()[0]).split("/")[0]]
record["verb_class"] = list(labels).index(
    (row["description"].split()[0]).split("/")[0]
)"""

fps = 22
offset = 5  # seconds
labels = [
    "Spread jelly on a bread slice",
    "Slice a potato",
    "Get/replace items from refrigerator/cabinets/drawers",
    "Clean a plate with a towel",
    "Pour water from a pitcher into a glass",
    "Stack on table: 3 each large/small plates, bowls",
    "Spread almond butter on a bread slice",
    "Slice a cucumber",
    "Clean a pan with a sponge",
    "Load dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils",
    "Open/close a jar of almond butter",
    "Slice bread",
    "Peel a cucumber",
    "Clean a plate with a sponge",
    "Clear cutting board",
    "Set table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils",
    "Clean a pan with a towel",
    "Peel a potato",
    "Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils",
    "Get items from cabinets: 3 each large/small plates, bowls, mugs, glasses, sets of utensils",
]
uid = random.randint(0, 20000)
dataset = []
timestamps = []
timestamps_int = []


def get_formatted_timestamp(timestamp):
    return float(timestamp.strip().replace(" ", "."))


def get_math_floor(float_value):
    return math.floor(float(float_value) * 10) / 10


def generate_record(index, row, cnt_video_split, start_frame=None, stop_frame=None):
    record = {}
    record["uid"] = uid + index
    record["participant_id"] = "S04"
    record["video_id"] = f"S04_0{cnt_video_split}"
    record["narration"] = row["description"]
    record["verb"] = row["description"]
    record["verb_class"] = labels.index(row["description"])

    if (start_frame is None and stop_frame is None) and cnt_video_split == 1:
        record["start_timestamp"] = float(row["start"])
        record["stop_timestamp"] = float(row["stop"])
        try:
            if get_math_floor(float(row["start"])) in timestamps[0]:
                index_start = timestamps[0].index(get_math_floor(float(row["start"])))
            else:
                index_start = timestamps_int[0].index(int(float(row["start"])))

            if get_math_floor(float(row["stop"])) in timestamps[0]:
                index_stop = timestamps[0].index(get_math_floor(float(row["stop"])))
            else:
                index_stop = timestamps_int[0].index(int(float(row["stop"])))

            record["start_frame"] = index_start + 1
            record["stop_frame"] = index_stop + 1
        except:
            return "Timestamps not found"
    else:
        record["start_timestamp"] = float(timestamps[cnt_video_split - 1][start_frame])
        record["stop_timestamp"] = float(timestamps[cnt_video_split - 1][stop_frame])
        record["start_frame"] = start_frame
        record["stop_frame"] = stop_frame

    return record


def dataset_augmentation(record):
    duration = int(record["stop_timestamp"] - record["start_timestamp"])
    if duration < (offset * 2):
        return [record]

    records = []
    next = 0
    first_iteration = True
    while first_iteration or new_record["stop_frame"] != record["stop_frame"]:
        first_iteration = False
        new_record = record.copy()

        new_record["start_timestamp"] = new_record["start_timestamp"] + next
        new_record["stop_timestamp"] = new_record["start_timestamp"] + offset

        new_record["start_frame"] = (
            new_record["start_frame"] + (fps * next) - (21 if next > 0 else 0)
        )
        new_record["stop_frame"] = new_record["start_frame"] + (fps * offset)

        next += offset + 1
        if record["start_timestamp"] + next > record["stop_timestamp"]:
            new_record["stop_timestamp"] = record["stop_timestamp"]
            new_record["stop_frame"] = record["stop_frame"]

        records.append(new_record)

    return records


def rgb_action_net_creation(out_path):
    data = get_data_from_pkl_pd("action-net/S04_1")
    uid_offset = 0

    for i in range(1, 6):
        with open(f"action-net/C0{i}_timestamps.txt", "r") as file:
            lines = file.readlines()
            timestamps.append(
                [get_math_floor(get_formatted_timestamp(line)) for line in lines]
            )
            timestamps_int.append(
                [int(get_formatted_timestamp(line)) for line in lines]
            )

    for index, row in data.iterrows():
        if index == 0:
            continue

        record = generate_record(uid_offset, row, 1)
        start_frame = record["start_frame"]
        stop_frame = record["stop_frame"]

        # Dataset augmentation by splitting video
        records = dataset_augmentation(record)
        for r in records:
            uid_offset += 1
            # Adding record(s) to dataset
            dataset.append(r)

        for k in range(2, 6):  # 4 = numeber of cameras not aligned
            if (row["description"].split()[0]).split("/")[0] == "Get" and k == 3:
                continue  # avoid to append record of camera 3 for Get/Put verb where no action is performed/viewable

            record = generate_record(uid_offset, row, k, start_frame, stop_frame)

            # Dataset augmentation by splitting video
            records = dataset_augmentation(record)
            for r in records:
                uid_offset += 1
                # Adding record(s) to dataset
                dataset.append(r)

    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(dataset)

    # Split dataset into train and validation
    train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)

    # Save numpy array to .pkl file
    with open(f"{out_path}_train.pkl", "wb") as file:
        pickle.dump(train_data, file)
    
    with open(f"{out_path}_test.pkl", "wb") as file:
        pickle.dump(val_data, file)

    print("RGB Action Net Creation: done")
    return df
