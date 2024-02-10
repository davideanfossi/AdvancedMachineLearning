import numpy as np
from utils.temporal_aggregation import aggregate_features
from utils.extract_pkl import *

pkl_folder = [
    "saved_features/aggregated_test_D5",
    "saved_features/aggregated_test_D10",
    "saved_features/aggregated_test_D25",
    "saved_features/aggregated_test_U5",
    "saved_features/aggregated_test_U10",
    "saved_features/aggregated_test_U25",
]

pkl_folder_action_net = [
    "action-net/ActionNet_test",
    "action-net/ActionNet_train"
]

if __name__ == "__main__":
    extract_pkl_pd(pkl_folder_action_net)
    """
    aggregate_features("train")
    extract_pkl(pkl_folder)

    data = np.array((get_data_from_pkl("saved_features/test_D5"))['features'])
    data_aggregated = np.array((get_data_from_pkl("saved_features/aggregated_test_D5"))['features'])

    print(f"\nData shape: {data.shape}")
    print(f"Data aggregated shape: {data_aggregated.shape}")
    """
