############
#
# Copyright (c) 2022 MIT CSAIL and Joseph DelPreto
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
# IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# See https://action-net.csail.mit.edu for more usage information.
# Created 2021-2022 for the MIT ActionNet project by Joseph DelPreto [https://josephdelpreto.com].
#
############

import h5py
import numpy as np
from scipy import interpolate  # for the resampling example
import sys

# NOTE: HDFView is a helpful program for exploring HDF5 contents.
#   The official download page is at https://www.hdfgroup.org/downloads/hdfview.
#   It can also be downloaded without an account from https://www.softpedia.com/get/Others/Miscellaneous/HDFView.shtml.


class HDF5Extractor:
    def __init__(
        self, in_path, out_folder=None, mode="sensor", device="myo-left", stream="emg"
    ):
        self.in_path = in_path
        self.out_folder = out_folder
        self.mode = mode
        self.save = False
        self.device = device
        self.stream = stream
        self.h5_file = h5py.File(in_path, "r")

    def extract_hdf5(self, save=False):
        self.save = save

        match self.mode:
            case "sensor":
                self.extract_sensor_data()
            case "label":
                self.extract_label_data()
            case "sensor_label":
                self.extract_sensor_data_for_one_label()
            case "resample":
                self.resample_sensor_data()

    def get_sensor_data(self, device_name, stream_name):
        # Get the data as an Nx8 matrix where each row is a timestamp and each column is an EMG channel.
        emg_data = self.h5_file[device_name][stream_name]["data"]
        emg_data = np.array(emg_data)
        # Get the timestamps for each row as seconds since epoch.
        emg_time_s = self.h5_file[device_name][stream_name]["time_s"]
        # squeeze (optional) converts from a list of single-element lists to a 1D list
        emg_time_s = np.squeeze(np.array(emg_time_s))
        # Get the timestamps for each row as human-readable strings.
        emg_time_str = self.h5_file[device_name][stream_name]["time_str"]
        # squeeze (optional) converts from a list of single-element lists to a 1D list
        emg_time_str = np.squeeze(np.array(emg_time_str))

        return emg_data, emg_time_s, emg_time_str

    def get_activities_data(self, device_name, stream_name):
        # Get the timestamped label data.
        # As described in the HDF5 metadata, each row has entries for ['Activity', 'Start/Stop', 'Valid', 'Notes'].
        activity_datas = self.h5_file[device_name][stream_name]["data"]
        activity_times_s = self.h5_file[device_name][stream_name]["time_s"]
        # squeeze (optional) converts from a list of single-element lists to a 1D list
        activity_times_s = np.squeeze(np.array(activity_times_s))
        # Convert to strings for convenience.
        activity_datas = [
            [x.decode("utf-8") for x in datas] for datas in activity_datas
        ]

        # Combine start/stop rows to single activity entries with start/stop times.
        #   Each row is either the start or stop of the label.
        #   The notes and ratings fields are the same for the start/stop rows of the label, so only need to check one.

        # some activities may have been marked as 'Bad' or 'Maybe' by the experimenter; submitted notes with the activity typically give more information
        exclude_bad_labels = True
        activities_labels = []
        activities_start_times_s = []
        activities_end_times_s = []
        activities_ratings = []
        activities_notes = []
        for row_index, time_s in enumerate(activity_times_s):
            label = activity_datas[row_index][0]
            is_start = activity_datas[row_index][1] == "Start"
            is_stop = activity_datas[row_index][1] == "Stop"
            rating = activity_datas[row_index][2]
            notes = activity_datas[row_index][3]
            if exclude_bad_labels and rating in ["Bad", "Maybe"]:
                continue
            # Record the start of a new activity.
            if is_start:
                activities_labels.append(label)
                activities_start_times_s.append(time_s)
                activities_ratings.append(rating)
                activities_notes.append(notes)
            # Record the end of the previous activity.
            if is_stop:
                activities_end_times_s.append(time_s)

        return (
            activities_labels,
            activities_start_times_s,
            activities_end_times_s,
            activities_ratings,
            activities_notes,
        )

    def extract_sensor_data(self):
        ####################################################
        # Example of reading sensor data: read Myo EMG data.
        ####################################################
        print()
        print("=" * 65)
        print("Extracting EMG data from the HDF5 file")
        print("=" * 65)

        device_name = self.device  # e.g = 'myo-left'
        stream_name = self.stream  # e.g = 'emg'

        emg_data, emg_time_s, emg_time_str = self.get_sensor_data(
            device_name, stream_name
        )

        print("EMG Data:")
        print(" Shape", emg_data.shape)
        print(" Preview:")
        print(emg_data)
        print()
        print("EMG Timestamps")
        print(" Shape", emg_time_s.shape)
        print(" Preview:")
        print(emg_time_s)
        print()
        print("EMG Timestamps as Strings")
        print(" Shape", emg_time_str.shape)
        print(" Preview:")
        print(emg_time_str)
        print()

        if self.out_folder is not None and self.save:
            with open(f"{self.out_folder}/emg_data.txt", "w") as txt_file:
                txt_file.write(str(emg_data))

            with open(f"{self.out_folder}/emg_time_s.txt", "w") as txt_file:
                txt_file.write(str(emg_time_s))

            with open(f"{self.out_folder}/emg_time_str.txt", "w") as txt_file:
                txt_file.write(str(emg_time_str))

    def extract_label_data(self):
        ####################################################
        # Example of reading label data
        ####################################################
        print()
        print("=" * 65)
        print("Extracting activity labels from the HDF5 file")
        print("=" * 65)

        device_name = self.device  # e.g = 'experiment-activities'
        stream_name = self.stream  # e.g = 'activities'

        activities_labels, activities_start_times_s, activities_end_times_s, _, _ = (
            self.get_activities_data(device_name, stream_name)
        )

        print("Activity Labels:")
        print(activities_labels)
        print()
        print("Activity Start Times")
        print(activities_start_times_s)
        print()
        print("Activity End Times")
        print(activities_end_times_s)

        if self.out_folder is not None and self.save:
            with open(f"{self.out_folder}/activities_labels.txt", "w") as txt_file:
                txt_file.write(str(activities_labels))

            with open(
                f"{self.out_folder}/activities_start_times_s.txt", "w"
            ) as txt_file:
                txt_file.write(str(activities_start_times_s))

            with open(f"{self.out_folder}/activities_end_times_s.txt", "w") as txt_file:
                txt_file.write(str(activities_end_times_s))

    def extract_sensor_data_for_one_label(self):
        ####################################################
        # Example of getting sensor data for a label.
        ####################################################
        print()
        print("=" * 65)
        print("Extracting EMG data during a specific activity")
        print("=" * 65)

        device_name = self.device  # e.g = 'experiment-activities'
        stream_name = self.stream  # e.g = 'activities'

        emg_data, emg_time_s, emg_time_str = self.get_sensor_data(
            self.device, self.stream
        )
        activities_labels, activities_start_times_s, activities_end_times_s, _, _ = (
            self.get_activities_data(device_name, stream_name)
        )

        # Get EMG data for the first instance of the second label.
        target_label = activities_labels[1]
        target_label_instance = 0

        # Find the start/end times associated with all instances of this label.
        label_start_times_s = [
            t
            for (i, t) in enumerate(activities_start_times_s)
            if activities_labels[i] == target_label
        ]

        label_end_times_s = [
            t
            for (i, t) in enumerate(activities_end_times_s)
            if activities_labels[i] == target_label
        ]

        # Only look at one instance for now.
        label_start_time_s = label_start_times_s[target_label_instance]
        label_end_time_s = label_end_times_s[target_label_instance]

        # Segment the data!
        emg_indexes_forLabel = np.where(
            (emg_time_s >= label_start_time_s) & (emg_time_s <= label_end_time_s)
        )[0]
        emg_data_forLabel = emg_data[emg_indexes_forLabel, :]
        emg_time_s_forLabel = emg_time_s[emg_indexes_forLabel]
        emg_time_str_forLabel = emg_time_str[emg_indexes_forLabel]

        print(
            'EMG Data for Instance %d of Label "%s"'
            % (target_label_instance, target_label)
        )
        print()
        print("Label instance start time  :", label_start_time_s)
        print("Label instance end time    :", label_end_time_s)
        print("Label instance duration [s]:", (label_end_time_s - label_start_time_s))
        print()
        print("EMG data during instance:")
        print(" Shape:", emg_data_forLabel.shape)
        print(" Preview:", emg_data_forLabel)
        print()
        print("EMG timestamps during instance:")
        print(" Shape:", emg_time_s_forLabel.shape)
        print(" Preview:", emg_time_s_forLabel)
        print()
        print("EMG timestamps as strings during instance:")
        print(" Shape:", emg_time_str_forLabel.shape)
        print(" Preview:", emg_time_str_forLabel)

        if self.out_folder is not None and self.save:
            with open(f"{self.out_folder}/emg_data_forLabel.txt", "w") as txt_file:
                txt_file.write(str(emg_data_forLabel))

            with open(f"{self.out_folder}/emg_time_s_forLabel.txt", "w") as txt_file:
                txt_file.write(str(emg_time_s_forLabel))

            with open(f"{self.out_folder}/emg_time_str_forLabel.txt", "w") as txt_file:
                txt_file.write(str(emg_time_str_forLabel))

    def resample_sensor_data(self):
        ####################################################
        # Example of resampling data so segmented lengths
        #  can match across sensors with different rates.
        # Note that the below example resamples the entire
        #  data, but it could also be applied to individual
        #  extracted segments if desired.
        ####################################################
        print()
        print("=" * 65)
        print("Resampling segmented IMU data to match the EMG sampling rate")
        print("=" * 65)

        emg_data, emg_time_s, _ = self.get_sensor_data(self.device, self.stream)

        # Get acceleration data.
        device_name = self.device  # e.g. = 'myo-left'
        stream_name = self.stream  # e.g. = 'acceleration_g'

        # Get the data as an Nx3 matrix where each row is a timestamp and each column is an acceleration axis.
        acceleration_data = np.array(self.h5_file[device_name][stream_name]["data"])

        # Get the timestamps for each row as seconds since epoch.
        acceleration_time_s = np.squeeze(
            np.array(self.h5_file[device_name][stream_name]["time_s"])
        )  # squeeze (optional) converts from a list of single-element lists to a 1D list

        # Get the timestamps for each row as human-readable strings.
        acceleration_time_str = np.squeeze(
            np.array(self.h5_file[device_name][stream_name]["time_str"])
        )  # squeeze (optional) converts from a list of single-element lists to a 1D list

        # Resample the acceleration to match the EMG timestamps.
        #  Note that the IMU streamed at about 50 Hz while the EMG streamed at about 200 Hz.
        fn_interpolate_acceleration = interpolate.interp1d(
            acceleration_time_s,  # x values
            acceleration_data,  # y values
            axis=0,  # axis of the data along which to interpolate
            kind="linear",  # interpolation method, such as 'linear', 'zero', 'nearest', 'quadratic', 'cubic', etc.
            fill_value="extrapolate",  # how to handle x values outside the original range
        )
        acceleration_time_s_resampled = emg_time_s
        acceleration_data_resampled = fn_interpolate_acceleration(
            acceleration_time_s_resampled
        )

        sampling_rate = (emg_data.shape[0] - 1) / (max(emg_time_s) - min(emg_time_s))
        sampling_rate_acc = (acceleration_data.shape[0] - 1) / (
            max(acceleration_time_s) - min(acceleration_time_s)
        )
        sampling_rate_resemaple = (acceleration_data_resampled.shape[0] - 1) / (
            max(acceleration_time_s_resampled) - min(acceleration_time_s_resampled)
        )

        print("EMG Data:")
        print(" Shape", emg_data.shape)
        print(" Sampling rate: %0.2f Hz" % sampling_rate)
        print()
        print("Acceleration Data Original:")
        print(" Shape", acceleration_data.shape)
        print(" Sampling rate: %0.2f Hz" % sampling_rate_acc)
        print()
        print("Acceleration Data Resampled to EMG Timestamps:")
        print(" Shape", acceleration_data_resampled.shape)
        print(" Sampling rate: %0.2f Hz" % sampling_rate_resemaple)
        print()

        if self.out_folder is not None and self.save:
            with open(
                f"{self.out_folder}/acceleration_data_resampled.txt", "w"
            ) as txt_file:
                txt_file.write(str(acceleration_data_resampled))

            with open(
                f"{self.out_folder}/acceleration_time_s_resampled.txt", "w"
            ) as txt_file:
                txt_file.write(str(acceleration_time_s_resampled))

            with open(
                f"{self.out_folder}/acceleration_time_str_resampled.txt", "w"
            ) as txt_file:
                txt_file.write(str(acceleration_time_str))
