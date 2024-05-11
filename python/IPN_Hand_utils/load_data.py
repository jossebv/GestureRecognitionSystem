import pandas as pd
import numpy as np
import os

class IPNHandLandmarks:
    def __init__(self, source, full_video=False):
        self.source = source
        self.full_video = full_video
        
    def get_data_path(self, video_name):
        '''
            Gets the path of the annotations of a video.

            Params:
            -source: Source folder with the video landmarks
            -video_name: Name of the video

            Returns:
            -annotations_path
        '''
        return os.path.join(self.source, f"{video_name}_poses_landamarks.csv")
    
    def __load_landmarks_from_file(self, data_path, first_frame=0, last_frame=0):
        '''
            Gets the landmarks of a video.

            Params:
            -annotations_path: Path of the annotations
            -first_frame: First frame of the gesture
            -last_frame: Last frame of the gesture

            Returns:
            -annotations_list
        '''
        landmarks = pd.read_csv(data_path).to_numpy()

        if last_frame == 0:
            last_frame = landmarks.shape[0]

        return landmarks[first_frame:last_frame, :42]
        


    def __load_annotations_full_video(self, video_name):
        '''
            Gets the name of a video and returns an array of landmarks of the video.

            Params:
            -video_name: Name of the video
            -source: Source folder with the video landmarks

            Returns:
            -annotations_list
        '''
        #Get the annotations
        data_path = self.get_data_path(video_name)
        landamarks = self.__load_landmarks_from_file(data_path)
        return landamarks
    
    def __load_annotations_gesture(self, annotations_list):
        '''
            Gets the annotations of a gestyre and returns an array of landmarks of that gesture.

            Params:
            -source: Source folder with the video landmarks
            -annotations_list: List of the annotations of the gesture

            Returns:
            -annotations_list
        '''
        #Get the annotations
        video_name = annotations_list[0]
        last_frame = annotations_list[4] # To have the last frame included
        if self.source.split("/")[-1] == "pose_features_windowed":
            first_frame = annotations_list[3]
        else:
            first_frame = annotations_list[3]-1 # To have the first frame included

        data_path = self.get_data_path(video_name)
        landamarks = self.__load_landmarks_from_file(data_path, first_frame, last_frame)
        return landamarks
    
    def get_landmarks(self, annotations_list):
        '''
            Gets the landmarks of a video.

            Params:
            -video_name: Name of the video

            Returns:
            -annotations_list
        '''
        video_name = annotations_list[0]

        if self.full_video:
            return self.__load_annotations_full_video(video_name)
        else:
            return self.__load_annotations_gesture(annotations_list)
