import os

import numpy as np
import cv2
import mediapipe as mp
import pandas as pd
from datetime import datetime
from tqdm import tqdm


def extract_XY(recognizer, IMAGE_FILES, images_user_path, out_path, columns):
    # Create the array to store the frames landmarks
    extracted_landmarks = np.empty((len(IMAGE_FILES), 42))

    # Process the images

    for idx, file in enumerate(IMAGE_FILES):
        path_img = os.path.join(images_user_path, file)
        mp_image = mp.Image.create_from_file(path_img)
        results = recognizer.recognize(mp_image)

        values = np.empty(42)
        if len(results.hand_landmarks) == 0:
            if idx == 0:
                values = np.array([0 for _ in range(42)])
            else:
                values = np.array([np.nan for _ in range(42)])
        else:
            for i, norm_landmark in enumerate(results.hand_landmarks[0]):
                x = norm_landmark.x
                y = 1 - norm_landmark.y
                values[i * 2 : ((i * 2) + 2)] = [x, y]

        # Save the values on the landmarks array
        extracted_landmarks[idx] = values

    # Create the dataframe
    video_df = pd.DataFrame(extracted_landmarks, columns=columns)
    # Interpolate the missing values
    video_df = video_df.interpolate(method="linear", axis=0)
    # Save the dataframe
    video_df.to_csv(out_path, sep=",", header=True, index=False)


if __name__ == "__main__":
    root_path = "/mnt/RESOURCES/josemanuelbravo/GestureRecognitionSystem/data/IPN_Hand/frames/all_frames"

    columns = [["x" + str(i), "y" + str(i)] for i in range(21)]
    columns = [item for sublist in columns for item in sublist]

    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = GestureRecognizerOptions(
        base_options=BaseOptions(
            model_asset_path="figure_creation/gesture_recognizer.task"
        ),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )

    for user in tqdm(os.listdir(root_path), desc="Users processed"):
        if user == ".DS_Store":
            continue
        images_user_path = os.path.join(root_path, user)
        IMAGE_FILES = sorted(os.listdir(images_user_path))
        directory = "../features/IPN_Hand/pose_features_w_interp/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        out_path_df = os.path.join(directory + user + "_poses_landamarks.csv")

        recognizer = GestureRecognizer.create_from_options(options)

        extract_XY(recognizer, IMAGE_FILES, images_user_path, out_path_df, columns)
