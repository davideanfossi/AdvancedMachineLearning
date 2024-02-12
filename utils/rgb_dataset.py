import random
import pickle
import pandas as pd
import numpy as np
from utils.extract_pkl import *

labels = {
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
dataset = []
uid = random.randint(0, 20000)
timestamps = []


def rgb_action_net_creation(out_path):
    data = get_data_from_pkl_pd("train_val_an/S04_1")
    cnt_video_split = 1

    for i in range(1, 5):
        with open(f"train_val_an/C0{i}_timestamps.txt", "r") as file:
            timestamps.append(
                ([round(float(line.strip().replace(" ", "."))) for line in file])
            )

    for index, row in data.iterrows():
        if index == 0:
            continue
        if index % 10 == 0:
            cnt_video_split += 1

        # Access individual columns using row[column_name]
        record = {}
        record["uid"] = uid + index
        record["participant_id"] = "S04"
        record["video_id"] = f"S04_0{cnt_video_split}"
        record["narration"] = row["description"]
        record["start_timestamp"] = float(row["start"])
        record["stop_timestamp"] = float(row["stop"])

        try:
            record["start_frame"] = timestamps[0].index(round(float(row["start"])))
            record["stop_frame"] = timestamps[0].index(round(float(row["stop"])))
        except:
            record["start_frame"] = 0
            record["stop_frame"] = 0

        record["verb"] = labels[(row["description"].split()[0]).split("/")[0]]
        record["verb_class"] = list(labels).index(
            (row["description"].split()[0]).split("/")[0]
        )

        dataset.append(record)

    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(dataset)

    # Save numpy array to .pkl file
    with open(f"{out_path}.pkl", "wb") as file:
        pickle.dump(df, file)

    print("RGB Action Net Creation: done")
    return df
