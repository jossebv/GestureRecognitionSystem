"""
# Select Data Module

From the loaded data select the data for training and testing

## Methods: 
- get_train_test_users_lists
- get_train_test_index_data_new_df
"""

import pandas as pd
import numpy as np


def get_train_test_users_lists():
    """
    Obtains lists of train and test users

    Returns
    -------
    - videos_train_list
        List with the users that will be used for training the model
    - videos_test_list
        List with the users that will be used for testing the model
    """
    videos_train_list = pd.read_csv(
        "../data/IPN_Hand/annotations/Video_TrainList.txt", sep='\t', header=None).loc[:, 0].tolist()
    for i in range(len(videos_train_list)):
        videos_train_list[i] = videos_train_list[i].split('#')[0]
    videos_train_list = np.unique(videos_train_list)

    videos_test_list = pd.read_csv(
        "../data/IPN_Hand/annotations/Video_TestList.txt", sep='\t', header=None).loc[:, 0].tolist()
    for i in range(len(videos_test_list)):
        videos_test_list[i] = videos_test_list[i].split('#')[0]
    videos_test_list = np.unique(videos_test_list)
    return [videos_train_list, videos_test_list]


def get_train_test_index_data_new_df(df, data, videos_train_list, videos_test_list):
    """
    Selects train and test data from original data

    Returns
    -------
    - train_index:
        List with all training videos index
    - test_index:
        List with all testing videos index
    - train_data:
        NDarray with the data for training
    - test_data:
        NDarray with the data for testing
    - train_df:
        df with all the training videos
    - test_df: 
        df with all the testing videos
    """

    # TODO: posiblemente puedo hacer todas la iteraciones en un solo bucle, mejorando la velocidad
    df["user"] = df["video"]
    for i in range(len(df)):
        df.loc[i, "user"] = df["video"][i].split('#')[0]

    train_index = []
    test_index = []
    for i in range(len(df)):
        if (df["user"][i] in videos_train_list):
            train_index.append(i)
        elif (df["user"][i] in videos_test_list):
            test_index.append(i)

    train_df = df.loc[df["user"] == videos_train_list[0]]
    for i in range(1, len(videos_train_list)):
        current_train_df = df.loc[df["user"] == videos_train_list[i]]
        train_df = pd.concat([train_df, current_train_df])

    test_df = df.loc[df["user"] == videos_test_list[0]]
    for i in range(1, len(videos_test_list)):
        current_test_df = df.loc[df["user"] == videos_test_list[i]]
        test_df = pd.concat([test_df, current_test_df])

    train_data = data[train_index, :, :]
    test_data = data[test_index, :, :]

    return [train_index, test_index, train_data, test_data, train_df, test_df]
