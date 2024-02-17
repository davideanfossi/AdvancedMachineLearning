import random
import pickle
import statistics
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

fps = 30
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

labels_remapping = {
    "Spread jelly on a bread slice": (0, "Spread"),
    "Slice a potato": (1, "Slice"),
    "Get/replace items from refrigerator/cabinets/drawers": (2, "Get/Put"),
    "Clean a plate with a towel": (3, "Clean"),
    "Pour water from a pitcher into a glass": (4, "Pour"),
    "Stack on table: 3 each large/small plates, bowls": (5, "Stack"),
    "Spread almond butter on a bread slice": (0, "Spread"),
    "Slice a cucumber": (1, "Slice"),
    "Clean a pan with a sponge": (3, "Clean"),
    "Load dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils": (
        6,
        "Load",
    ),
    "Open/close a jar of almond butter": (7, "Open/Close"),
    "Slice bread": (1, "Slice"),
    "Peel a cucumber": (1, "Slice"),
    "Clean a plate with a sponge": (3, "Clean"),
    "Clear cutting board": (3, "Clean"),
    "Set table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils": (
        5,
        "Stack",
    ),
    "Clean a pan with a towel": (3, "Clean"),
    "Peel a potato": (1, "Slice"),
    "Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils": (
        6,
        "Load",
    ),
    "Get items from cabinets: 3 each large/small plates, bowls, mugs, glasses, sets of utensils": (
        2,
        "Get/Put",
    ),
}

uid = random.randint(0, 20000)
dataset = []
dataset_reduced = []
dataset_emg = []
timestamps = []
timestamps_int = []


def generate_record(index, row, first_frame, cnt=1):
    record = {}
    record["uid"] = uid + index
    record["participant_id"] = "P04"
    record["video_id"] = f"P04_0{cnt}"
    record["narration"] = row["description"]
    record["verb"] = row["description"]
    record["verb_class"] = labels.index(row["description"])
    record["start_timestamp"] = float(row["start"])
    record["stop_timestamp"] = float(row["stop"])
    record["start_frame"] = round((float(row["start"]) - first_frame) * fps)
    record["stop_frame"] = round((float(row["stop"]) - first_frame) * fps)
    record["myo_right_timestamps"] = row["myo_right_timestamps"]
    record["myo_left_timestamps"] = row["myo_left_timestamps"]
    record["myo_right_readings"] = row["myo_right_readings"]
    record["myo_left_readings"] = row["myo_left_readings"]

    return record


def dataset_augmentation(record, uid):
    emg_list = []
    duration = int(record["stop_timestamp"] - record["start_timestamp"])
    if duration < (offset * 2):
        return [record]

    records = []
    next = 0
    first_iteration = True
    while first_iteration or duration - next > offset:
        emg_record = {"left": [], "right": []}
        first_iteration = False
        new_record = record.copy()

        new_record["uid"] = uid
        new_record["start_timestamp"] = new_record["start_timestamp"] + next
        new_record["stop_timestamp"] = new_record["start_timestamp"] + offset

        new_record["start_frame"] = (
            new_record["start_frame"] + (fps * next) - (21 if next > 0 else 0)
        )
        new_record["stop_frame"] = new_record["start_frame"] + (fps * offset)

        next += offset + 1
        uid += 1
        if duration - next < offset:
            new_record["stop_timestamp"] = record["stop_timestamp"]
            new_record["stop_frame"] = record["stop_frame"]

        readings = []
        for i in range(len(new_record["myo_left_timestamps"])):
            ts = new_record["myo_left_timestamps"][i]
            if int(ts) >= int(new_record["start_timestamp"]) and int(ts) <= int(
                new_record["stop_timestamp"]
            ):
                readings.append(new_record["myo_left_readings"][i])
        new_record["myo_left_readings"] = readings

        readings = []
        for i in range(len(new_record["myo_right_timestamps"])):
            ts = new_record["myo_right_timestamps"][i]
            if int(ts) >= int(new_record["start_timestamp"]) and int(ts) <= int(
                new_record["stop_timestamp"]
            ):
                readings.append(new_record["myo_right_readings"][i])
        new_record["myo_right_readings"] = readings

        # print("LEFT - RIGHT")
        # print(len(emg_list["left"]), len(emg_list["right"]))
        # print("ABS")
        # print(abs(len(emg_list["left"]) - len(emg_list["right"])))
        emg_list.append(emg_list)
        records.append(new_record)

    # print("\n")
    # print(max([len(x["left"]) for x in l]))
    # print(min([len(x["left"]) for x in l]))
    # print(statistics.mean([len(x["left"]) for x in l]))

    # print(max([len(x["right"]) for x in l]))
    # print(min([len(x["right"]) for x in l]))
    # print(statistics.mean([len(x["right"]) for x in l]))
    return records


def remap_labels(record):
    record_reduced = record.copy()
    record_reduced["verb_class"] = labels_remapping[record["verb"]][0]
    record_reduced["verb"] = labels_remapping[record["verb"]][1]
    return record_reduced


def rgb_action_net_creation(out_path=None, out_path_reduced=None, out_path_emg=None):
    data = get_data_from_pkl_pd("action-net/S04_1")
    uid_offset = 0
    first_frame = 0

    for index, row in data.iterrows():
        if index == 0:
            first_frame = float(row["start"])
            continue

        record = generate_record(uid_offset, row, first_frame, 1)

        # Dataset augmentation by splitting video
        records = dataset_augmentation(record, uid_offset)
        uid_offset += len(records)
        for r in records:
            # Adding record(s) to dataset
            r_rgb = {
                "uid": r["uid"],
                "participant_id": r["participant_id"],
                "video_id": r["video_id"],
                "narration": r["narration"],
                "verb": r["verb"],
                "verb_class": r["verb_class"],
                "start_timestamp": r["start_timestamp"],
                "stop_timestamp": r["stop_timestamp"],
                "start_frame": r["start_frame"],
                "stop_frame": r["stop_frame"],
            }

            dataset.append(r_rgb)

            dataset_emg.append(
                {
                    "myo_right_timestamps": r["myo_right_timestamps"],
                    "myo_left_timestamps": r["myo_left_timestamps"],
                    "myo_right_readings": r["myo_right_readings"],
                    "myo_left_readings": r["myo_left_readings"],
                }
            )

            # Remap labels
            record_reduced = remap_labels(r_rgb)
            dataset_reduced.append(record_reduced)

    # Convert list of dictionaries to DataFrame
    df_rgb = pd.DataFrame(dataset)
    df_reduced = pd.DataFrame(dataset_reduced)
    df_emg = pd.DataFrame(dataset_emg)

    # Split dataset into train and validation
    train_data, val_data = train_test_split(df_rgb, test_size=0.2, random_state=42)
    train_data_emg, val_data_emg = train_test_split(
        df_emg, test_size=0.2, random_state=42
    )
    train_data_reduced, val_data_reduced = train_test_split(
        df_reduced, test_size=0.2, random_state=42
    )

    if out_path is not None:
        # Save numpy array to .pkl file
        with open(f"{out_path}_train.pkl", "wb") as file:
            pickle.dump(train_data, file)

        with open(f"{out_path}_test.pkl", "wb") as file:
            pickle.dump(val_data, file)

    if out_path_reduced is not None:
        # Save numpy array to .pkl file (reduced labels)
        with open(f"{out_path_reduced}_train.pkl", "wb") as file:
            pickle.dump(train_data_reduced, file)

        with open(f"{out_path_reduced}_test.pkl", "wb") as file:
            pickle.dump(val_data_reduced, file)

    if out_path_emg is not None:
        # Save numpy array to .pkl file
        with open(f"{out_path_emg}_train.pkl", "wb") as file:
            pickle.dump(train_data_emg, file)

        with open(f"{out_path_emg}_test.pkl", "wb") as file:
            pickle.dump(val_data_emg, file)

    print("RGB Action Net Creation: done")
    return df_rgb, df_reduced, df_emg
