import numpy as np
import pandas as pd
from scipy.signal import medfilt

# It downsamples each gesture (970 points) to final_points


def downsampling(data, final_points):
    new_data = np.zeros((data.shape[0], final_points, data.shape[2]))
    for i in range(len(data)):
        data_filtered = medfilt(
            data[i, :, :], [int(np.ceil(final_points) // 2 * 2 + 1), 1])
        data_downsampled = data_filtered[0::int(data.shape[1]/final_points), :]
        new_data[i, :, :] = data_downsampled[0:final_points, :]
    return new_data

# It downsamples each gesture (970 points) to final_points using a different approach based on number of real frames from gesture


def downsampling2(data, final_points):
    new_data = np.zeros((data.shape[0], final_points, data.shape[2]))
    num_frames = pd.read_csv(
        "../data/IPN_Hand/annotations/Annot_List.txt")["frames"]
    for i in range(len(data)):
        if (num_frames[i] <= final_points):
            new_data[i, :, :] = np.pad(data[i, data.shape[1]-num_frames[i]:, :], ((
                final_points-num_frames[i], 0), (0, 0)), 'constant', constant_values=0)
        else:
            relevant_data = data[i, data.shape[1]-num_frames[i]:, :]
            data_filtered = medfilt(
                relevant_data, [int(final_points // 2 * 2 + 1), 1])
            data_downsampled = data_filtered[0::int(
                relevant_data.shape[0]/final_points), :]
            new_data[i, :, :] = data_downsampled[0:final_points, :]
    return new_data
