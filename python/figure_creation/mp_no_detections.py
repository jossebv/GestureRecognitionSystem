import os

import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm


def extract_XY(
    recognizer,
    video_name,
    IMAGE_FILES,
    images_user_path,
    annotations,
    not_found_dataframe,
):
    for idx, file in enumerate(IMAGE_FILES):
        path_img = os.path.join(images_user_path, file)
        mp_image = mp.Image.create_from_file(path_img)
        results = recognizer.recognize(mp_image)

        if len(results.hand_landmarks) == 0:
            gest_annotations = annotations[(annotations["video_name"] == video_name)]
            gest_annotations = gest_annotations[
                gest_annotations["first_frame"] <= idx + 1
            ]
            gest_annotations = gest_annotations[
                gest_annotations["last_frame"] >= idx + 1
            ]

            not_found_dataframe.loc[len(not_found_dataframe["video_name"])] = {
                "video_name": video_name,
                "label": gest_annotations["label"].values[0],
                "frame": idx + 1,
                "path": path_img,
            }

    return not_found_dataframe


if __name__ == "__main__":
    root_path = "/mnt/RESOURCES/josemanuelbravo/GestureRecognitionSystem/data/IPN_Hand"
    annotations_path = os.path.join(root_path, "annotations", "Annot_TrainList.txt")

    results = pd.DataFrame(columns=["video_name", "label", "frame", "path"])

    annotations = pd.read_csv(annotations_path)
    video_names = pd.unique(annotations["video_name"])

    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path="gesture_recognizer.task"),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )

    for video in tqdm(video_names, desc="Processing videos:"):
        images_user_path = os.path.join(root_path, "frames/all_frames", video)
        IMAGE_FILES = sorted(os.listdir(images_user_path))

        recognizer = GestureRecognizer.create_from_options(options)

        results = extract_XY(
            recognizer, video, IMAGE_FILES, images_user_path, annotations, results
        )

        results.to_csv("no_detections.csv", index=False)
