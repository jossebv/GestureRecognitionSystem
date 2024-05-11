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
INPUT_SHAPE = (100, 42, 1)
INTERPOLATION = 0
NUM_OFFICIAL_CLASSES = 13
NUM_CLASSES = 14
NORMALIZATION = "FF"  # Accepted values: "FF", "F0", "NONE"

DATA = "pose_features_w_interp"
DATA_PATH = f"../features/IPN_Hand/{DATA}"

if INTERPOLATION != 0:
    DATA_PATH = f"{DATA_PATH}_interp_{INTERPOLATION}"

GAD_ACTIVE = False
GAD_MODE = "PEAK"  # Accepted values: "MORE_ENERGY", "THRESHOLD", "PEAK"

# OPTIMIZER
LEARNING_RATE = 1e-4

# TRAINING
EPOCHS = 60

# TUBELET EMBEDDING
PATCH_SIZE = (5, 42)
STRIDE = (5, 42)

# ViViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 64
NUM_LAYERS = 4
NUM_HEADS = 4
DROPOUT_RATE = 0.1

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


# Tubelet Embedding
class TubeletEmbedding(layers.Layer):
    """
    Tubelet embedding layer. It projects the video into a sequence of patches.

    Params:
        -embed_dim: Embedding dimension
        -patch_size: Size of the patch
    """

    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
        )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos):
        projected_patches = self.projection(videos)
        flattened_patches = self.flatten(projected_patches)
        return flattened_patches


# Positional Embedding


class PositionalEncoder(layers.Layer):
    """
    Positional encoder layer. It encodes the position of the patches of the video.

    Params:
        -embed_dim: Embedding dimension
    """

    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim
        )
        self.positions = tf.range(start=0, limit=num_tokens, delta=1)

    def call(self, encoded_tokens):
        # Encode the positions and add it to the encoded tokens
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens


# Video Vision Transformer Full Model


def create_vivit_classifier(
    tubelet_embedder,
    positional_encoder,
    input_shape=INPUT_SHAPE,
    transformer_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    embed_dim=PROJECTION_DIM,
    layer_norm_eps=LAYER_NORM_EPS,
    num_classes=NUM_CLASSES,
):
    """
    Creates the ViViT model.

    Params:
        -tubelet_embedder: TubeletEmbedding layer
        -positional_encoder: PositionalEncoder layer
        -input_shape: Shape of the input
        -transformer_layers: Number of transformer layers
        -num_heads: Number of heads of the MultiHeadAttention layer
        -embed_dim: Embedding dimension
        -layer_norm_eps: Epsilon of the LayerNormalization layer
        -num_classes: Number of classes to classify

    Returns:
        -model: The ViViT model
    """
    # Get the input layer
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    patches = tubelet_embedder(inputs)
    # Encode patches.
    encoded_patches = positional_encoder(patches)

    # Added by me
    encoded_patches = layers.Dropout(DROPOUT_RATE)(encoded_patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization and MHSA
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output, attention_scores = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=DROPOUT_RATE
        )(x1, x1, return_attention_scores=True)

        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential(
            [
                layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu),
                layers.Dense(units=embed_dim, activation=tf.nn.gelu),
            ]
        )(x3)

        # Added by me
        x3 = layers.Dropout(DROPOUT_RATE)(x3)

        # Skip connection
        encoded_patches = layers.Add()([x3, x2])

    # Layer normalization and Global average pooling.
    representation = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
    representation = layers.GlobalAvgPool1D()(representation)

    # Classify outputs.
    outputs = layers.Dense(units=num_classes, activation="softmax")(representation)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


#  Training
def run_experiment():
    # Initialize a W&B run
    run = wandb.init(
        project="ViViT_landmarks_experiments",
        notes="",
        group="video_length",
        job_type="training",
        name="100_GAD_DEACTIVATED",
        entity="josebravopacheco-team",
        config={
            "epochs": EPOCHS,
            "data": DATA,
            "batch_size": BATCH_SIZE,
            "patch_size": PATCH_SIZE,
            "stride": STRIDE,
            "learning_rate": LEARNING_RATE,
            "projection_dim": PROJECTION_DIM,
            "input_shape": INPUT_SHAPE,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "normalization": NORMALIZATION,
            "dropout_rate": DROPOUT_RATE,
            "interpolation": INTERPOLATION,
            "gad_active": GAD_ACTIVE,
            "gad_mode": GAD_MODE,
        },
    )

    # PATCH_SIZE = wandb.config.patch_size

    # Initialize model
    model = create_vivit_classifier(
        tubelet_embedder=TubeletEmbedding(
            embed_dim=PROJECTION_DIM, patch_size=PATCH_SIZE
        ),
        positional_encoder=PositionalEncoder(embed_dim=PROJECTION_DIM),
        input_shape=INPUT_SHAPE,
        transformer_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        embed_dim=PROJECTION_DIM,
        layer_norm_eps=LAYER_NORM_EPS,
    )

    # Compile the model with the optimizer, loss function
    # and the metrics.
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    # Train the model.
    _ = model.fit(
        trainloader,
        epochs=EPOCHS,
        validation_data=testloader,
        callbacks=[WandbMetricsLogger()],
    )

    _, accuracy, top_5_accuracy = model.evaluate(testloader)

    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    # Save the model
    savedir = f"../models/IPN_Hand/patch-{PATCH_SIZE}/projection_dim-{PROJECTION_DIM}/num_heads-{NUM_HEADS}/num_layers-{NUM_LAYERS}/dropout_rate-{DROPOUT_RATE}/norm-{NORMALIZATION}/interp-{INTERPOLATION}"
    os.makedirs(savedir, exist_ok=True)
    model.save(f"{savedir}/model.keras")

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

    return model


model = run_experiment()


# SWEEP CONFIG
# sweep_id = wandb.sweep(sweep=sweep_config, project="ViViT_landmarks")
# wandb.agent(sweep_id, function=run_experiment)
