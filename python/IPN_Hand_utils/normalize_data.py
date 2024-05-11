"""
# Normalization module

It contains the methods used for normalizing the landmark data

## Methods
- normalize_from_0_landmark
- normalize_from_first_frame
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from IPN_Hand_utils.generator import get_gesture_landmarks_from_full_video_landmarks


def __normalize_from_0_landmark(frames: np.ndarray):
    '''
        Moves the hand so the wrist of each frame is at (0,0)
    '''
    for landmark in frames:
        if (landmark[0] + landmark[1] != 0):
            x_center = landmark[0]
            y_center = landmark[1]
            if (landmark.sum() != 0):
                for i in range(0,frames.shape[1],2):
                    landmark[i] = landmark[i] - x_center
                    landmark[i+1] = landmark[i+1] - y_center
    return frames

def __normalize_from_first_frame(frames: np.ndarray):
    '''
        Moves the hand of all frames so the wrist of the first frame is at (0,0)
    '''
    if (frames[0,0] + frames[0,1] != 0):
        x_center = frames[0,0]
        y_center = frames[0,1]
        for landmarks in frames:
            for i in range(0,frames.shape[1],2):
                if (landmarks.sum() != 0):
                    landmarks[i] = landmarks[i] - x_center
                    landmarks[i+1] = landmarks[i+1] - y_center
    return frames


def __normalize_video_landmarks(video_annotations, norm_type):

    '''
        Gets the annotation list of a video and returns an array of landmarks of the video normalized.

        Params:
         -video_annotations: List with the information of the video: [video_name, label_name, label, first_frame, last_frame, number_frames]
         -norm_type: Type of normalization to apply

        Returns:
         -norm_landmarks
    '''

    video_name = video_annotations[0,0]
    video_landmarks = pd.read_csv(f"../features/IPN_Hand/pose_features_windowed/{video_name}_poses_landamarks.csv").to_numpy()

    norm_landmarks = None

    for gesture_annotation in video_annotations:

        gest_landmarks = get_gesture_landmarks_from_full_video_landmarks(video_landmarks, gesture_annotation)

        if norm_type == "F0":
            norm_gest_landmarks = __normalize_from_0_landmark(gest_landmarks)
        elif norm_type == "FF":
            norm_gest_landmarks = __normalize_from_first_frame(gest_landmarks)

        norm_landmarks = np.concatenate((norm_landmarks, norm_gest_landmarks), axis=0) if norm_landmarks is not None else norm_gest_landmarks

    return norm_landmarks

def __save_landmarks(landmarks, video_name):
    """
        Saves the landmarks in a csv file

        Params:
            -landmarks: Landmarks to save
            -video_name: Name of the video
    """
    columns = [["x" + str(i), "y" + str(i)] for i in range(21)]
    columns = [item for sublist in columns for item in sublist]

    df = pd.DataFrame(landmarks, columns=columns)
    df.to_csv(f"../features/IPN_Hand/pose_features_windowed/{video_name}_poses_landamarks.csv", index=False)


# Normalization API

def normalize_landmarks(norm_type: str):
    """
        Gets the data from the pose_features_windowed folder and normalizes it

        Params:
            -norm_type: Type of normalization to apply
    """

    print(f"\033[94m Normalization:\033[00m Normalizing data with {norm_type} normalization...")
    # Get the annotation list
    annotations = pd.read_csv(
        "../features/IPN_Hand/pose_features_windowed/Annot_List.txt").to_numpy()
    
    video_names = np.unique(annotations[:,0])
    
    for video in tqdm(video_names, desc="Normalizing videos"):
        video_annotations = annotations[annotations[:,0] == video]
        norm_landmarks = __normalize_video_landmarks(video_annotations, norm_type)
        __save_landmarks(norm_landmarks, video)


    return 0