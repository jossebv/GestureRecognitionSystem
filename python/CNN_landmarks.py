#!/usr/bin/env python
# coding: utf-8

#  Imports
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras import layers
import wandb
from wandb.keras import WandbMetricsLogger
from sklearn.metrics import confusion_matrix
from IPN_Hand_utils.dataset_windowing import make_data_windows, WindowSettings
from IPN_Hand_utils.generator import IPN_Hand_generator
from IPN_Hand_utils.normalize_data import normalize_landmarks

# Setting seed for reproducibitility
SEED = 42
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
keras.utils.set_random_seed(SEED)

wandb.login(key="33abf58c3769a66af5da5304a87811c8b98b7cfd")


# Hyperparameters
# DATA
AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 16
INPUT_SHAPE = (101, 42, 1)
INTERPOLATION = 0
NUM_OFFICIAL_CLASSES = 13
NUM_CLASSES = 14
NORMALIZATION = "NONE"  # Accepted values: "FF", "F0", "NONE"

DATA = "pose_features_w_interp"
DATA_PATH = f"../features/IPN_Hand/{DATA}"

if INTERPOLATION != 0:
    DATA_PATH = f"{DATA_PATH}_interp_{INTERPOLATION}"

GAD_ACTIVE = True
GAD_MODE = "PEAK"  # Accepted values: "MORE_ENERGY", "THRESHOLD", "PEAK"

# OPTIMIZER
LEARNING_RATE = 1e-4

# TRAINING
EPOCHS = 60

# sweep_config = {
#   "name" : "vivit_hyperparam_search",
#   "method" : "grid",
#   "parameters" : {
#     "patch_size" : {
#         "values" : [(4,42), (4,8)]
#     }
#   }
# }


# IPN_Hand Loader


@tf.function
def preprocess(landmarks: tf.Tensor, labels: tf.Tensor):
    """
    Preprocess the landmarks and the labels. It adds a dimension to the landmarks to help the convolutional layer.

    Params:
     -landmarks: Tensor with the landmarks
     -labels: Tensor with the labels

    Returns:
     -(landmarks, labels)
    """
    landmarks = landmarks[..., tf.newaxis]
    return (landmarks, labels)


def prepare_dataloader(
    annotations_path, N=0, batch_size=BATCH_SIZE, loader_type="train"
):
    """
    Prepares the dataloader for the dataset.

    Params:
     -annotations_path: Path to the annotations file
     -N: Number of annotations to get. If 0, it gets all the annotations. Default is 0.
     -batch_size: Batch size
     -loader_type: Type of the loader. Accepted values: "train", "test". Default is "train".

    Returns:
     -dataloader: Dataloader of the dataset
    """

    dataset = tf.data.Dataset.from_generator(
        IPN_Hand_generator,
        args=[annotations_path],
        output_types=(np.float32, np.float32),
        output_shapes=((INPUT_SHAPE[0], INPUT_SHAPE[1]), ()),
    )

    if loader_type == "train":
        dataset = dataset.shuffle(BATCH_SIZE * 2)

    dataloader = (
        dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return dataloader


# Preprocess the data
window_settings = WindowSettings(
    data_path=DATA_PATH,
    interpolation=INTERPOLATION,
    window_size=INPUT_SHAPE[0],
    gad_active=GAD_ACTIVE,
    gad_mode=GAD_MODE,
    normalization=NORMALIZATION,
)
make_data_windows(window_settings)
if NORMALIZATION != "NONE":
    normalize_landmarks(NORMALIZATION)

# Prepare the dataloaders
trainloader = prepare_dataloader(
    "../features/IPN_Hand/pose_features_windowed/Annot_TrainList.txt",
    loader_type="train",
)
testloader = prepare_dataloader(
    "../features/IPN_Hand/pose_features_windowed/Annot_TestList.txt", loader_type="test"
)


if __name__ == "__main__":
    wandb.init(
        project="ViViT_landmarks_experiments",
        notes="",
        group="CNN",
        job_type="training",
        name="ResNet50",
        entity="josebravopacheco-team",
        config={
            "batch_size": BATCH_SIZE,
            "input_shape": INPUT_SHAPE,
            "interpolation": INTERPOLATION,
            "num_classes": NUM_CLASSES,
            "normalization": NORMALIZATION,
            "data": DATA,
            "data_path": DATA_PATH,
            "gad_active": GAD_ACTIVE,
            "gad_mode": GAD_MODE,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
        },
    )

    model = keras.Sequential(
        [
            keras.applications.ResNet50(
                include_top=False,
                weights=None,
                input_shape=INPUT_SHAPE,
                pooling="avg",
            ),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5_accuracy"),
        ],
    )

    _ = model.fit(
        trainloader,
        validation_data=testloader,
        epochs=EPOCHS,
        callbacks=[WandbMetricsLogger()],
    )

    _, acc, top_5_acc = model.evaluate(testloader)
    print(f"Accuracy: {acc}")
    print(f"Top-5 Accuracy: {top_5_acc}")

    # Confusion matrix
    yS = np.concatenate([y for _, y in testloader], axis=0)
    predictionS = np.argmax(model.predict(testloader), axis=1)
    matrix = confusion_matrix(yS, predictionS, labels=np.arange(0, NUM_CLASSES))
    print(matrix.sum(), ": Number of testing examples")
    print(matrix)
    confusion_dataframe = pd.DataFrame(matrix)
    confusion_table = wandb.Table(dataframe=confusion_dataframe)
    wandb.log({"confusion_matrix": confusion_table})

    wandb.finish()
