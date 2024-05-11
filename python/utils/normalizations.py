"""
# Normalization module

It contains the methods used for normalizing the landmark data

## Methods
- normalize_from_0_landmark
- normalize_from_first_frame
"""

import numpy as np


def normalize_from_0_landmark(data: np.ndarray):
    """
    # Normalize_from_0_landmark
    Normalizes each frame using as reference the wrist landmark of that frame
    """
    new_data = np.copy(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if ((data[i, j, 0] + data[i, j, 1]) != 0):
                x_center = data[i, j, 0]
                y_center = data[i, j, 1]
                for k in range(2, data.shape[2], 2):
                    new_data[i, j, k] = data[i, j, k] - x_center
                    new_data[i, j, k+1] = data[i, j, k+1] - y_center
                # new_data[i,j,0] = data[i,j,0]-x_center
                # new_data[i,j,1] = data[i,j,1]-y_center
    return new_data


def normalize_from_first_frame(data: np.ndarray):
    """
    # Normalize_from_first_frame
    Normalizes each frame using as reference the wrist landmark of the first frame
    """
    new_data = np.copy(data)
    for i in range(data.shape[0]):
        if (data[i, :, :].sum() != 0):
            l = 0
            while ((data[i, l, 0] + data[i, l, 1]) == 0):
                l = l+1
            x_center = data[i, l, 0]
            y_center = data[i, l, 1]
            for j in range(data.shape[1]):
                if ((data[i, j, 0] + data[i, j, 1]) != 0):
                    for k in range(0, data.shape[2], 2):
                        new_data[i, j, k] = data[i, j, k] - x_center
                        new_data[i, j, k+1] = data[i, j, k+1] - y_center
                    # new_data[i,j,0] = data[i,j,0]-x_center
                    # new_data[i,j,1] = data[i,j,1]-y_center
    return new_data
