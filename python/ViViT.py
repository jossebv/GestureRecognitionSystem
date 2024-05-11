#!/usr/bin/env python
# coding: utf-8

# # Imports
# 

# In[42]:


import os
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from scipy.signal import medfilt
from sklearn.metrics import confusion_matrix
import wandb
from wandb.keras import WandbMetricsLogger


#Setting seed for reproducibitility
SEED = 42
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
keras.utils.set_random_seed(SEED)

wandb.login(key="33abf58c3769a66af5da5304a87811c8b98b7cfd")


# # Hyperparameters
# 

# In[43]:


# DATA
DATASET_NAME = "organmnist3d"
BATCH_SIZE = 12
AUTO = tf.data.AUTOTUNE
FRAME_SIZE = (240,320)
INPUT_SHAPE = INPUT_SIZE = (140, FRAME_SIZE[0], FRAME_SIZE[1], 1)
NUM_OFFICIAL_CLASSES = 13
NUM_CLASSES = 14

# OPTIMIZER
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# TRAINING
EPOCHS = 60

# TUBELET EMBEDDING
PATCH_SIZE = (5,80,80)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2

# ViViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 128 
NUM_HEADS = 8
NUM_LAYERS = 8
DROPOUT_RATE = 0.1


# # IPN_Hand Loader

# In[45]:


@tf.function
def preprocess(frames: tf.Tensor, label: tf.Tensor):
    """Preprocess the frames tensors and parse the labels."""
    # Preprocess images
    frames = tf.image.convert_image_dtype(
        frames[
            ..., tf.newaxis
        ],  # The new axis is to help for further processing with Conv3D layers
        tf.float16,
    )
    # Parse label
    label = tf.cast(label, tf.float16)

    return frames, label

def generate_video(annotations):
    video_name = annotations[0]
    label = annotations[2]-1
    first_frame = annotations[3]
    last_frame = annotations[4]
    number_frames = annotations[5]
    video_path = os.path.join("../features/IPN_Hand/coordinates_images/all_frames/", video_name)
    video_arr = np.zeros((number_frames,INPUT_SHAPE[1],INPUT_SHAPE[2]), dtype=np.uint8)
    for frame_number in range(first_frame, last_frame+1):
        frame_number_formatted = str(frame_number).rjust(6, '0')
        image_path = os.path.join(video_path, f"{video_name}_{frame_number_formatted}_landmarks.jpg")
        image = cv2.imread(image_path)[:,:,0]
        video_arr[frame_number-first_frame] = image

    if (number_frames > INPUT_SHAPE[0]):
        video_arr = video_arr[0:INPUT_SHAPE[0],:]
    else:
        video_arr = np.pad(video_arr, ((0,INPUT_SHAPE[0]-video_arr.shape[0]),(0,0),(0,0)), "constant", constant_values=0)

    return video_arr,label

def dataGenerator(annotations_path, N):
    annotations_list = pd.read_csv(annotations_path.decode("utf-8"))

    if (NUM_OFFICIAL_CLASSES == 13):
        annotations_list = annotations_list[annotations_list["label"] != 1]

    annotations_list = annotations_list.to_numpy()
    
    video_index = 0
    if N == 0:
        N = len(annotations_list)
    
    while video_index < N:
        video,label = generate_video(annotations_list[video_index])
        yield (video, label)
        video_index = video_index + 1


def prepare_dataloader(annotations_path, N=0, batch_size = BATCH_SIZE, loader_type="train"):

    dataset = tf.data.Dataset.from_generator(dataGenerator, args=[annotations_path, N], output_types=(np.uint8, np.uint8), output_shapes=(INPUT_SHAPE[0:3],()))

    if loader_type == "train":
        dataset = dataset.shuffle(BATCH_SIZE * 2)

    dataloader = (
        dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return dataloader

trainloader = prepare_dataloader("../data/IPN_Hand/annotations/Annot_TrainList.txt", loader_type="train")
testloader = prepare_dataloader("../data/IPN_Hand/annotations/Annot_TestList.txt", loader_type="test")


# # Tubelet Embedding

# In[46]:


class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(
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


# # Positional Embedding

# In[47]:


class PositionalEncoder(layers.Layer):
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


# # Video Vision Transformer Full Model
# 

# In[48]:


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
    # Get the input layer
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    patches = tubelet_embedder(inputs)
    # Encode patches.
    encoded_patches = positional_encoder(patches)

    #Added by me
    encoded_patches = layers.Dropout(DROPOUT_RATE)(encoded_patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization and MHSA
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=DROPOUT_RATE
        )(x1, x1)

        #Added by me
        attention_output = layers.Dropout(DROPOUT_RATE)(attention_output)

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

        # Skip connection
        encoded_patches = layers.Add()([x3, x2])

        #Added by me
        encoded_patches = layers.Dropout(DROPOUT_RATE)(encoded_patches)

    # Layer normalization and Global average pooling.
    representation = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
    representation = layers.GlobalAvgPool1D()(representation)

    # Classify outputs.
    outputs = layers.Dense(units=num_classes, activation="softmax")(representation)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# # Train

# In[49]:


def run_experiment():

    run = wandb.init(
        project="ViViT",
        notes="Using constant 0 padding for videos with less than 140 frames.",
        config={
            "epochs" : EPOCHS,
            "batch_size" : BATCH_SIZE,
            "learning_rate" : LEARNING_RATE,
            "projection_dim" : PROJECTION_DIM,
            "num_heads" : NUM_HEADS,
            "num_layers" : NUM_LAYERS,
            "patch_size" : PATCH_SIZE,
            "input_shape" : INPUT_SHAPE,
            "dropout_rate" : DROPOUT_RATE,
        }
    )

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
        layer_norm_eps=LAYER_NORM_EPS
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
    _ = model.fit(trainloader, validation_data=testloader, epochs=EPOCHS, callbacks=[WandbMetricsLogger()])

    _, accuracy, top_5_accuracy = model.evaluate(testloader)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    #Confusion Matrix.
    yS = np.concatenate([y for _,y in testloader], axis=0) 
    predictionS = np.argmax(model.predict(testloader), axis=1) 
    matrix = confusion_matrix(yS, predictionS, labels=np.arange(0, NUM_CLASSES))
    print(matrix.sum(), ': Number of testing examples')
    print(matrix)


    return model


model = run_experiment()


# In[ ]:




