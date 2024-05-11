import os
import pandas as pd
import numpy as np


def __get_gesture_landmarks_and_label(annotations_list):
    '''
        Gets the annotation list of a video and returns a tuple with the array of landmarks and the label of the video.

        Params:
         -annotations_list: List with the information of the video: [video_name, label_name, label, first_frame, last_frame, number_frames]
         -gesture_video_lenght: Number of frames of the gesture video
         -num_official_classes: Number of classes of the dataset

        Returns:
         -(landmarks, label)
    '''
    video_name = annotations_list[0]
    label = annotations_list[2] - 1 # To have them between 0 and 13 instead of 1 and 14
    first_frame = annotations_list[3]
    last_frame = annotations_list[4]

    #Recover the landmarks from the csv files
    csv_path = os.path.join(f"../features/IPN_Hand/pose_features_windowed/",f"{video_name}_poses_landamarks.csv")
    full_video_landmarks = pd.read_csv(csv_path).to_numpy()

    #Get only the gesture frames
    landmarks = full_video_landmarks[first_frame:last_frame,0:42]

    #Parse the landmarks  
    landmarks = landmarks.astype(np.float32)

    return (landmarks, label)




def get_gesture_landmarks_from_full_video_landmarks(full_video_landmarks ,annotations_list):
    '''
        Gets the annotation list of a video and returns an array of landmarks of the video.

        Params:
         -full_video_landmarks: Array with the landmarks of the video
         -annotations_list: List with the information of the video: [video_name, label_name, label, first_frame, last_frame, number_frames]

        Returns:
         -landmarks
    '''
    first_frame = annotations_list[3]
    last_frame = annotations_list[4]

    #Get only the gesture frames
    landmarks = full_video_landmarks[first_frame:last_frame,0:42]

    #Parse the landmarks  
    landmarks = landmarks.astype(np.float32)

    return landmarks




def __get_annotations(annotations_path, num_official_classes):
    '''
        Gets the first N annotations of the annotations file.

        Params:
         -annotations_path: Path to the annotations file
         -N: Number of annotations to get

        Returns:
         -annotations: Numpy ndarray with the annotations
    '''
    annotations = pd.read_csv(annotations_path.decode("utf-8"))

    if num_official_classes == 13:
        annotations = annotations[(annotations["label"] != 1)]
        
    return annotations.to_numpy()


def IPN_Hand_generator(annotations_path, num_official_classes = 13):
    '''
        Generator that yields the landmarks and the label of a video.

        Params:
            -annotations_path: Path to the annotations file
            -N: Number of annotations to get
        
        Yields:
            -(landmarks, label)
    '''
    annotations_list = __get_annotations(annotations_path, num_official_classes)

    gesture_video_index = 0
    while gesture_video_index < len(annotations_list):
        landmarks,label = __get_gesture_landmarks_and_label(annotations_list[gesture_video_index])
        yield (landmarks, label)
        gesture_video_index = gesture_video_index + 1