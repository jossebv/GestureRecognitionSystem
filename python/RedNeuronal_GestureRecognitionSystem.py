import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import LSTM
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.utils import class_weight
import pandas as pd
import sys

from utils import loadData, normalizations, selectTrainTestData, windows
from utils.downsamplers import downsampling, downsampling2

from sliding_window import sliding_window


np.random.seed(2022)

####################################################

output_path = sys.argv[1]
output_file = sys.argv[1] + "output.out"
#sys.stdout = open(output_file, "w")
approach = sys.argv[2]  # "Selection"
network_type = sys.argv[3]  # "LSTM"
final_points = int(sys.argv[4])  # 50
num_oficial_classes = int(sys.argv[5])  # 14
num_classes = 14
norm_type = sys.argv[6]  # L0
epochs = int(sys.argv[7])
batch_size = int(sys.argv[8])

# direct values
# output_path = "/"
# output_file = "output.out"
# sys.stdout=open(output_file,"w")
# approach = "Selection"
# network_type = "LSTM"
# final_points = 50
# num_oficial_classes = 14
# num_classes = 14
# norm_type = "FF"
# epochs = 20
# batch_size = 100

num_channels = 42
dropout = 0.4
cnn_num = 16

####################################################

print("Loading data...")

# loadData.load_data()
# loadData.create_numpy_with_data_and_df_with_video_label()

new_df = pd.read_csv("../features/IPN_Hand/to_use/video_label.csv")
data = np.load('../features/IPN_Hand/to_use/data.npy')
num_timesteps = data.shape[1]

print("Data loaded.")


print("Selecting data...")

[videos_train_list, videos_test_list] = selectTrainTestData.get_train_test_users_lists()

# final_points=100
# [train_index, test_index, x_train, x_test, train_df, test_df] = get_train_test_index_data_new_df(new_df, downsampling(data, final_points), videos_train_list, videos_test_list)
# num_timesteps=x_train.shape[1]

# final_points=50
# [train_index, test_index, x_train, x_test, train_df, test_df] = get_train_test_index_data_new_df(new_df, downsampling2(data, final_points), videos_train_list, videos_test_list)
# num_timesteps=x_train.shape[1]


print("Normalizing, selecting labels...")

if norm_type == "None":
    normalized_data = data
elif norm_type == "L0":
    normalized_data = normalizations.normalize_from_0_landmark(data)
elif norm_type == "FF":
    normalized_data = normalizations.normalize_from_first_frame(data)

data = normalized_data


if ((approach == "Selection_Subwindows") | (approach == "Selection_Subwindows_FFT")):
    data = windows.selection_last_N(data, final_points)


# To manage subwindows in each gesture instance
if ((approach == "Subwindows") | (approach == "Subwindows_FFT") | (approach == "Selection_Subwindows") | (approach == "Selection_Subwindows_FFT")):
    print("Into subwindows...")
    ws = 10
    ss = 5
    num_subwindows = int((data.shape[1]-ws)/ss+1)

    all_data = np.zeros((len(data), num_subwindows, ws, data.shape[2]))
    for i in range(len(data)):
        current_gesture = data[i]
        data_subwindows = sliding_window(current_gesture, (ws, 42), (ss, 42))
        all_data[i] = data_subwindows
    data = all_data

if ((approach == "Subwindows") | (approach == "Selection_Subwindows")):
    data = np.reshape(all_data, ((
        all_data.shape[0], all_data.shape[1], all_data.shape[2] *
        all_data.shape[3]
    )))
elif ((approach == "Subwindows_FFT") | (approach == "Selection_Subwindows_FFT")):
    all_data = np.zeros((len(data), num_subwindows, int(ws/2), data.shape[3]))
    for i in range(len(data)):
        for j in range(len(data[i])):
            for k in range(data.shape[2]):
                current_window = data[i][j]
                current_window_FFT = np.absolute(np.fft.fft(
                    current_window, current_window.shape[1], 1, None))
                all_data[i][j] = current_window_FFT[0:int(ws/2), :]
    data = np.reshape(all_data, ((
        all_data.shape[0], all_data.shape[1], all_data.shape[2]*all_data.shape[3])))


if (num_oficial_classes == 13):
    new_df3 = new_df[(new_df["label"] != 1)]
    new_df3 = new_df3.reset_index()
    data3 = data[(new_df["label"] != 1)]

    data = data3
    new_df = new_df3


if ((approach == "None") | (approach == "Subwindows") | (approach == "Subwindows_FFT") | (approach == "Selection_Subwindows") | (approach == "Selection_Subwindows_FFT")):
    new_data = data
elif (approach == "Selection"):
    new_data = windows.selection_last_N(data, final_points)
elif (approach == "Downsampling"):
    new_data = downsampling2(data, final_points)
elif (approach == "DownsamplingSelection"):
    new_data = windows.selection_last_N(downsampling2(
        data, final_points*2), final_points)


[train_index, test_index, x_train, x_test, train_df, test_df] = selectTrainTestData.get_train_test_index_data_new_df(
    new_df, new_data, videos_train_list, videos_test_list)
num_timesteps = x_train.shape[1]

x_train2 = x_train
x_test2 = x_test


# To create one-hot enconding of labels
if (num_oficial_classes == 14):
    y_train = np.array(train_df["label"])
    y_test = np.array(test_df["label"])

    y_train_format = np.zeros((y_train.shape[0], num_classes), dtype=int)
    for j in range(y_train.shape[0]):
        y_train_format[j, int(y_train[j])-1] = 1
    y_test_format = np.zeros((y_test.shape[0], num_classes), dtype=int)
    for j in range(y_test.shape[0]):
        y_test_format[j, int(y_test[j])-1] = 1
elif (num_oficial_classes == 13):
    y_train2 = np.array(train_df[(train_df["label"] != 1)]["label"])
    y_test2 = np.array(test_df[(test_df["label"] != 1)]["label"])

    y_train_format = np.zeros((y_train2.shape[0], num_classes), dtype=int)
    for j in range(y_train2.shape[0]):
        y_train_format[j, int(y_train2[j])-1] = 1
    y_test_format = np.zeros((y_test2.shape[0], num_classes), dtype=int)
    for j in range(y_test2.shape[0]):
        y_test_format[j, int(y_test2[j])-1] = 1


print("Data selected.")


# Selection of type of network
if (network_type == "LSTM"):
    model = Sequential()
    model.add(LSTM(32, return_sequences=False,
              input_shape=(num_timesteps, 42)))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))
elif (network_type == "CNN"):
    x_train2 = np.reshape(
        x_train2, (x_train2.shape[0], x_train2.shape[1], x_train2.shape[2], 1))
    x_test2 = np.reshape(
        x_test2, (x_test2.shape[0], x_test2.shape[1], x_test2.shape[2], 1))

    model = Sequential()
    model.add(Conv2D(cnn_num, (1, 10), padding='same', input_shape=(
        final_points, num_channels, 1), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(1, 4)))
    model.add(Conv2D(cnn_num, (20,20), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(10,2)))
    # model.add(Dropout(dropout))
    # model.add(Conv2D(cnn_num, (1, 5),padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(1, 2)))
    # model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))
elif (network_type == "LSTM_Subwindows"):
    model = Sequential()
    model.add(LSTM(32, return_sequences=False,
              input_shape=(num_subwindows, ws*42)))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))
elif (network_type == "LSTM_Subwindows_FFT"):
    model = Sequential()
    model.add(LSTM(32, return_sequences=False,
              input_shape=(num_subwindows, int(ws/2)*42)))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',  metrics=['accuracy'])

# Train
history = model.fit(x_train2, y_train_format, batch_size=batch_size,
                    epochs=epochs, verbose=1, validation_split=0.0, shuffle=True)


# Test
print("---SETUP---")
print("Approach: ", approach)
print("Type of network: ", network_type)
print("Points per example: ", num_timesteps)
print("Num classes: ", num_oficial_classes)


prediction = model.predict(x_train2, batch_size=batch_size, verbose=1)
predictionS = np.argmax(prediction, axis=1) + 1
yS = np.argmax(y_train_format, axis=1) + 1


print("---TRAIN---")
matrix = confusion_matrix(yS, predictionS, labels=np.arange(1, num_classes+1))
print(matrix.sum(), ': Number of training examples')
print(matrix)
print(accuracy_score(yS, predictionS), ': Train accuracy')
print(f1_score(yS, predictionS, average='weighted'), ': Train fmeasure weighted')


print("---TEST---")
prediction = model.predict(x_test2, batch_size=batch_size, verbose=1)
predictionS = np.argmax(prediction, axis=1) + 1
yS = np.argmax(y_test_format, axis=1) + 1


matrix = confusion_matrix(yS, predictionS, labels=np.arange(1, num_classes+1))
print(matrix.sum(), ': Number of testing examples')
print(matrix)
print(accuracy_score(yS, predictionS), ': Test accuracy')
print(f1_score(yS, predictionS, average='weighted'), ': Test fmeasure weighted')

sys.stdout.close()
