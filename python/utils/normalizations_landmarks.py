import numpy as np
def normalize_from_0_landmark(frames: np.ndarray):
    '''
        Moves the hand so the wrist of each frame is at (0,0)
    '''
    for landmark in frames:
        if (landmark[0] + landmark[1] != 0):
            x_center = landmark[0]
            y_center = landmark[1]
            for i in range(0,frames.shape[1],2):
                if (landmark.sum() != 0):
                    landmark[i] = landmark[i] - x_center
                    landmark[i+1] = landmark[i+1] - y_center
    return frames

def normalize_from_first_frame(frames: np.ndarray):
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