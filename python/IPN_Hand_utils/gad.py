import numpy as np


def get_energy_between_frames(frame, prev_frame):
    """
    Gets the energy between two frames of a video. If one of the frames is 0, the energy is 0.

    Params:
     -frame: Array with the landmarks of the frame
     -prev_frame: Array with the landmarks of the previous frame

    Returns:
     -energy: The energy between the two frames
    """
    energy = 0
    if frame.sum() == 0 or prev_frame.sum() == 0:
        return 0

    for j in range(0, 42, 2):
        energy += np.sqrt(
            (frame[j] - prev_frame[j]) ** 2 + (frame[j + 1] - prev_frame[j + 1]) ** 2
        )

    return energy


def get_gesture_energy(landmarks: np.ndarray):
    """
    Gets the energy of a video.

    Params:
     -landmarks: Array with the landmarks of the video

    Returns:
     -gesture_energy: The energy of the video
    """
    gesture_energy = np.empty(landmarks.shape[0] - 1, dtype=np.float32)

    for i in range(0, landmarks.shape[0] - 1):
        if i == 0:
            energy = 0
        else:
            energy = get_energy_between_frames(landmarks[i, :], landmarks[i + 1, :])
        gesture_energy[i] = energy

    return gesture_energy


def get_gesture_window_with_more_energy(landmarks: np.ndarray, window_size=51, step=1):
    """
    Gets the gesture window of a video. The gesture window is the window with the highest energy of the video.

    Params:
     -landmarks: Array with the landmarks of the video
     -window_size: Size of the window to calculate the energy
     -step: Step to move the window. Default is 1

    Returns:
     -gesture_window: The gesture window of the video
    """
    gesture_window = landmarks[0:window_size, :]

    gesture_energy = get_gesture_energy(landmarks)

    energy = np.array(gesture_energy[0:window_size]).sum()
    max_energy = energy

    for i in range(1, len(gesture_energy) - (window_size - 1), step):
        energy = (
            energy
            - gesture_energy[i - step : i].sum()
            + gesture_energy[i + (window_size - step) : i + window_size].sum()
        )
        if energy > max_energy:
            max_energy = energy
            gesture_window = landmarks[i : i + window_size, :]

    return gesture_window


def get_gesture_window_from_energy_threshold(
    landmarks, window_size=51, energy_threshold=0.5
):
    """
    Gets the gesture window of a video. The gesture window is the window with length window_size centered in the first frame with energy higher than energy_threshold.

    Params:
     -landmarks: Array with the landmarks of the video
     -window_size: Size of the window to calculate the energy. It must be an odd number
     -energy_threshold: Threshold to consider a frame as a peak. Default is 0.5

    Returns:
        -gesture_window: The gesture window of the video
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be an odd number")

    gesture_window = landmarks[0:window_size, :]

    gesture_energy = get_gesture_energy(landmarks)

    for i in range(
        int(window_size / 2), len(gesture_energy) - (int(window_size / 2) - 1)
    ):
        if gesture_energy[i] > energy_threshold:
            gesture_window = landmarks[
                i - int(window_size / 2) : i + int(window_size / 2) + 1, :
            ]
            break

    return gesture_window


def get_gesture_window_from_peak(landmarks, window_size=51):
    """
    Gets the gesture window of a video. The gesture window is the window with length window_size centered in the absolute peak of the video.

    Params:
     -landmarks: Array with the landmarks of the video
     -window_size: Size of the window to calculate the energy. It must be an odd number

    Returns:
        -gesture_window: The gesture window of the video. Default is 51.
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be an odd number")

    gesture_window = landmarks[0:window_size, :]

    gesture_energy = get_gesture_energy(landmarks)
    max_energy = gesture_energy[0]

    for i in range(
        int(window_size / 2), len(gesture_energy) - (int(window_size / 2) - 1)
    ):
        energy = gesture_energy[i]
        if energy > max_energy:
            max_energy = energy
            gesture_window = landmarks[
                i - int(window_size / 2) : i + int(window_size / 2) + 1, :
            ]
    return gesture_window


def get_gesture_window(landmarks: np.ndarray, mode="FIRST_WINDOW", window_size=51):
    """
    Gets the gesture window of a video.

    This module has three modes:
        -MORE_ENERGY: The gesture window is the window with the highest energy of the video.
        -THRESHOLD: The gesture window is the window with length window_size centered in the FIRST frame transition with energy higher than energy_threshold.
        -PEAK: The gesture window is the window with length window_size centered in the absolute peak of the video.

    Params:
     -landmarks: Array with the landmarks of the video
     -mode: Mode to get the gesture window. It can be "FIRST_WINDOW" or "PEAK". Default is "FIRST_WINDOW"

    Returns:
     -gesture_window: The gesture window of the video
    """
    if mode == "MORE_ENERGY":
        return get_gesture_window_with_more_energy(landmarks, window_size)
    elif mode == "THRESHOLD":
        return get_gesture_window_from_energy_threshold(landmarks, window_size)
    elif mode == "PEAK":
        return get_gesture_window_from_peak(landmarks, window_size)
    else:
        raise ValueError("Mode must be 'MORE_ENERGY', 'FIRST_WINDOW' or 'PEAK'")
