"""
# LoadData module

Gets the landmarks previously processed, loads them adding the class column and create the numpy array with all the final data.

## Methods:
- load_data
- create_numpy_with_data_and_df_with_video_label
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_data():
    """Creates a .csv file of data including a label column"""
    videos_list = pd.read_csv("../data/IPN_Hand/videos_list.txt", header=None)
    df_annotations = pd.read_csv("../data/IPN_Hand/annotations/Annot_List.txt")

    # Load the first
    current_video = videos_list.loc[0][0]
    df = pd.read_csv("../features/IPN_Hand/pose_features/" +
                     current_video+"_poses_landamarks.csv")
    df['label'] = np.zeros(len(df))
    df_annotations_video = df_annotations[df_annotations["video"]
                                          == current_video]
    df_annotations_video = df_annotations_video.reset_index()

    for i in range(len(df_annotations_video)):
        df.loc[(df_annotations_video["t_start"][i] <= df["frame"]) & (
            df["frame"] <= df_annotations_video["t_end"][i]), "label"] = df_annotations_video["id"][i]

    # Load the rest
    for j in tqdm(range(1, len(videos_list)), desc="Labeling the videos..."):
        current_video = videos_list.loc[j][0]

        current_df = pd.read_csv(
            "../features/IPN_Hand/pose_features/"+current_video+"_poses_landamarks.csv")
        current_df['label'] = np.zeros(len(current_df))

        df_annotations_video = df_annotations[df_annotations["video"]
                                              == current_video]
        df_annotations_video = df_annotations_video.reset_index()

        for i in range(len(df_annotations_video)):
            current_df.loc[(df_annotations_video["t_start"][i] <= current_df["frame"]) & (
                current_df["frame"] <= df_annotations_video["t_end"][i]), "label"] = df_annotations_video["id"][i]

        df = pd.concat([df, current_df])

    df.to_csv(
        '../features/IPN_Hand/to_use/coordinates_frames_labelled.csv', index=False)


def create_numpy_with_data_and_df_with_video_label():
    """
        - It creates a numpy file with data and a .csv file with video and label information
        - Each gesture has 970 points and 42 features
        - 970 points because it is the max_length of a gesture instance (including 0s at the beginning)
        - If the gesture has less frames, then it is padded to has the same size
        - 42 features because we work with 21 landmarks with x and y coordinates
    """
    df = pd.read_csv(
        "../features/IPN_Hand/to_use/coordinates_frames_labelled.csv")
    num_points = 42
    max_length = pd.read_csv(
        "../data/IPN_Hand/annotations/Annot_List.txt")["frames"].max()
    num_total_examples = len(pd.read_csv(
        "../data/IPN_Hand/annotations/Annot_List.txt"))

    # new_df = pd.DataFrame([], columns=["data", "video", "label"])
    new_df = pd.DataFrame([], columns=["video", "label"])
    data = np.zeros((num_total_examples, max_length, num_points))
    df_annotations = pd.read_csv("../data/IPN_Hand/annotations/Annot_List.txt")
    fea_list = [["x" + str(j), "y" + str(j)] for j in range(int(num_points/2))]
    flat_fea_list = [item for sublist in fea_list for item in sublist]

    for i in tqdm(range(len(df_annotations)), desc="Padding the data..."):
        current_video = df_annotations["video"][i]
        current_label = df_annotations["id"][i]
        current_example = df.loc[(df["video"] == current_video) & (df["label"] == current_label) & (
            df_annotations["t_start"][i] <= df["frame"]) & (df["frame"] <= df_annotations["t_end"][i])]

        # 0s at the end
        # data[i] = np.pad(current_example[flat_fea_list], ((0, max_length-len(current_example[flat_fea_list])), (0, 0)), 'constant', constant_values=0)

        # 0s at the beginning
        data[i] = np.pad(current_example[flat_fea_list], ((
            max_length-len(current_example[flat_fea_list]), 0), (0, 0)), 'constant', constant_values=0)

        # 0s at both edges
        # num_zeros = max_length-len(current_example[flat_fea_list])
        # data[i] = np.pad(current_example[flat_fea_list], ((
        #     num_zeros/2, num_zeros/2), (0, 0)), 'constant', constant_values=0)

        # Repeat the sequence
        # data[i] = np.pad(current_example[flat_fea_list], ((
        #     0, max_length-len(current_example[flat_fea_list])), (0, 0)), "wrap")

        new_df.loc[i] = [current_video, current_label]

    new_df.to_csv('../features/IPN_Hand/to_use/video_label.csv', index=False)
    np.save("../features/IPN_Hand/to_use/data.npy", data)


__all__ = ["load_data", "create_numpy_with_data_and_df_with_video_label"]
