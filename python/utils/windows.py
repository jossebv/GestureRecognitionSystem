"""
# Windows module

Contains the functions for selecting a window from the data

## Methods:
- selection_last_N
"""

import numpy as np


def selection_last_N(data: np.ndarray, N: int):
    """
    # Selection_last_N
    It selects the last N points for each example (gesture)
    """
    new_data = np.zeros((data.shape[0], N, data.shape[2]))
    for i in range(len(data)):
        new_data[i, :, :] = data[i, data.shape[1]-N:, :]
    return new_data
