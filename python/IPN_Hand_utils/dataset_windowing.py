import pandas as pd
import numpy as np
from IPN_Hand_utils.gad import get_gesture_window
from tqdm import tqdm
import os

DATA_ROOT_PATH = "/mnt/RESOURCES/josemanuelbravo/GestureRecognitionSystem/data/IPN_Hand"


## CLASSES
class WindowSettings:
    def __init__(
        self, window_size, data_path, interpolation, gad_active, gad_mode, normalization
    ):
        self.window_size = window_size
        self.data_path = data_path
        self.interpolation = interpolation
        self.gad_active = gad_active
        self.gad_mode = gad_mode
        self.normalization = normalization


## FUNCTIONS
def is_gesture_in_annotation_list(gesture, annot_file):
    return (gesture == annot_file).all(axis=1).any()


def get_window_from_gesture_landmarks(gesture_landmarks, window_settings):
    window_size = window_settings.window_size
    gad_active = window_settings.gad_active
    gad_mode = window_settings.gad_mode

    if gesture_landmarks.shape[0] < window_size:
        gesture_landmarks = np.pad(
            gesture_landmarks,
            ((0, window_size - gesture_landmarks.shape[0]), (0, 0)),
            "constant",
            constant_values=0,
        )
    else:
        if gad_active:
            gesture_landmarks = get_gesture_window(
                gesture_landmarks, gad_mode, window_size
            )
        else:
            gesture_landmarks = gesture_landmarks[:window_size, :]
    return gesture_landmarks


def get_gesture_landmarks(gesture, window_settings):
    interpolation = window_settings.interpolation
    data_path = window_settings.data_path

    video_name = gesture[0]
    first_frame = (gesture[3] - 1) * (interpolation + 1)
    last_frame = (gesture[4] - 1) * (interpolation + 1)
    gesture_landmarks = pd.read_csv(
        f"{data_path}/{video_name}_poses_landamarks.csv"
    ).to_numpy()
    gesture_landmarks = gesture_landmarks[first_frame : last_frame + 1, :42]
    return gesture_landmarks


def update_annot_files(annot_files_to_update, gesture, i, window_size):
    annot_file_cp, annot_file_train, annot_file_test = annot_files_to_update
    if is_gesture_in_annotation_list(gesture, annot_file_train):
        annot_file_to_update = annot_file_train
    elif is_gesture_in_annotation_list(gesture, annot_file_test):
        annot_file_to_update = annot_file_test

    index = annot_file_to_update[(annot_file_to_update == gesture).all(1)].index[0]
    annot_file_to_update.loc[index, ["first_frame", "last_frame", "number_frames"]] = [
        i * window_size,
        (i + 1) * window_size,
        window_size,
    ]

    index = annot_file_cp[(annot_file_cp == gesture).all(1)].index[0]
    annot_file_cp.loc[index, ["first_frame", "last_frame", "number_frames"]] = [
        i * window_size,
        (i + 1) * window_size,
        window_size,
    ]

    return (annot_file_cp, annot_file_train, annot_file_test)


def get_video_windows(annot_file, annot_files_to_update, video_name, window_settings):
    window_size = window_settings.window_size

    video_annot_file = annot_file[annot_file["video_name"] == video_name]

    columns = [["x" + str(i), "y" + str(i)] for i in range(21)]
    columns = [item for sublist in columns for item in sublist]

    number_of_gestures = len(video_annot_file)

    # Create the array to store the frames landmarks
    extracted_landmarks = np.empty((number_of_gestures * window_size, 42))
    for i, gesture in enumerate(video_annot_file.to_numpy()):
        gesture_landmarks = get_gesture_landmarks(gesture, window_settings)
        gesture_landmarks_window = get_window_from_gesture_landmarks(
            gesture_landmarks, window_settings
        )
        extracted_landmarks[i * window_size : (i + 1) * window_size, :] = (
            gesture_landmarks_window
        )
        annot_files_to_update = update_annot_files(
            annot_files_to_update, gesture, i, window_size
        )

    extracted_landmarks_df = pd.DataFrame(extracted_landmarks, columns=columns)

    return extracted_landmarks_df, annot_files_to_update


def are_previous_windows_the_same(window_settings: WindowSettings):
    try:
        with open("../features/IPN_Hand/pose_features_windowed/settings.txt", "r") as f:
            previous_settings = eval(f.read())
            return previous_settings == window_settings.__dict__
    except FileNotFoundError:
        print(
            "\033[94m Windowing:\033[00m No previous windowing settings found. Windowing will be performed."
        )
        return False


def save_windows_settings(window_settings: WindowSettings):
    with open("../features/IPN_Hand/pose_features_windowed/settings.txt", "w") as f:
        f.write(str(window_settings.__dict__))


def get_annotations_files():
    annot_file = pd.read_csv(os.path.join(DATA_ROOT_PATH, "annotations/Annot_List.txt"))
    annot_file_train = pd.read_csv(
        os.path.join(DATA_ROOT_PATH, "annotations/Annot_TrainList.txt")
    )
    annot_file_test = pd.read_csv(
        os.path.join(DATA_ROOT_PATH, "annotations/Annot_TestList.txt")
    )
    return (annot_file, annot_file_train, annot_file_test)


def save_annotations_files_updated(annot_files_to_update):
    annot_file_cp, annot_file_train, annot_file_test = annot_files_to_update
    annot_file_cp.to_csv(
        "../features/IPN_Hand/pose_features_windowed/Annot_List.txt", index=False
    )
    annot_file_train.to_csv(
        "../features/IPN_Hand/pose_features_windowed/Annot_TrainList.txt", index=False
    )
    annot_file_test.to_csv(
        "../features/IPN_Hand/pose_features_windowed/Annot_TestList.txt", index=False
    )


def __print_windowing_info(window_settings: WindowSettings):
    print(f"\033[94m Windowing:\033[00m Starting windowing process. Settings are:")
    print(f"\tWindow size: {window_settings.window_size}")
    print(f"\tData path: {window_settings.data_path}")
    print(f"\tInterpolation: {window_settings.interpolation}")
    print(f"\tGAD active: {window_settings.gad_active}")
    print(f"\tGAD mode: {window_settings.gad_mode}")
    print(f"\tNormalization: {window_settings.normalization}")


def make_data_windows(window_settings: WindowSettings, force_windowing: bool = False):
    """
    Performs the windowing process on the dataset. The windowing process consists on extracting the landmarks of the gestures and storing them in a csv file.

    Parameters
    ----------
    window_settings: WindowSettings Object containing the windowing settings.
    force_windowing: bool If True, the windowing process will be performed even if previous windowing settings are found and match.

    Returns
    -------
    None
    """
    __print_windowing_info(window_settings)

    if are_previous_windows_the_same(window_settings) and not force_windowing:
        print(
            "\033[94m Windowing:\033[00m Previous windowing settings found and match. Windowing will not be performed."
        )
        return

    print(
        "\033[94m Windowing:\033[00m Previous windowing settings found but do not match or force_windowing is set to True. Windowing will be performed."
    )

    annot_file, annot_file_train, annot_file_test = get_annotations_files()
    annot_files_to_update = (annot_file.copy(), annot_file_train, annot_file_test)

    video_names = list(set(annot_file["video_name"]))  # Get the unique video names
    for video_name in tqdm(video_names, desc="\033[94m Windowing\033[00m"):
        video_landmarks_df, annot_files_to_update = get_video_windows(
            annot_file, annot_files_to_update, video_name, window_settings
        )
        video_landmarks_df.to_csv(
            f"../features/IPN_Hand/pose_features_windowed/{video_name}_poses_landamarks.csv",
            index=False,
        )

    save_annotations_files_updated(annot_files_to_update)
    save_windows_settings(window_settings)


os.makedirs("../features/IPN_Hand/pose_features_windowed", exist_ok=True)
