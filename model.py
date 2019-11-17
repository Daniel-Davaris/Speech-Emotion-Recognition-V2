import os
import random
import sys
import glob 
import h5py
import keras
import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly 
import plotly.graph_objs as go
import plotly.offline as py
import plotly.tools as tls
import seaborn as sns
import scipy.io.wavfile
import tensorflow as tf

# from tensorflow import keras
# from tensorflow.python.keras import backend as k
# from tensorflow.keras.models import Sequential
from keras import regularizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.callbacks import  History, ReduceLROnPlateau, CSVLogger

from keras.models import Sequential
from keras.models import Model

from keras.layers import Dense, Embedding, LSTM
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras import optimizers
from keras.optimizers import adam

from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm
from tensorflow.keras import backend
from clr_callback import CyclicLR
input_duration=3



# all transported files from data prep
tmp1 = pd.read_pickle("./dummy.pkl")   # pkl files as array is of differnt type
tmp3 = pd.read_pickle("./dummy.pkl")  # pkl files as array is of different type 
data3_df = pd.concat([tmp1, tmp3],ignore_index=True).reset_index(drop=True)
h5f = h5py.File('data.h5','r')
x_traincnn = h5f['one'][:]
x_testcnn = h5f['two'][:]
y_test = h5f['three'][:]
y_train = h5f['four'][:]

# lb = h5f['five'][:]
X_test = h5f['six'][:]
X_train = h5f['seven'][:]
# data3_df = h5f['last'][:]
h5f.close()




from keras import backend as K
# customer model untility function
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fscore(y_true, y_pred):
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    f_score = 2 * (p * r) / (p + r + K.epsilon())
    return f_score

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

# layers
model = Sequential()
model.add(Conv1D(256, 8, padding='same',input_shape=(X_train.shape[1],1)))
model.add(Activation('relu'))
model.add(Conv1D(256, 8, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(64, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(64, 8, padding='same'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(80))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(40))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('softmax'))
opt = keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)



model.summary()


model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', fscore])




# Model Training
# es = EarlyStopping(monitor='val_loss')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
# ts = keras.callbacks.tensorboard_v1.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
ts = keras.callbacks.TensorBoard(log_dir='./logs')
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=20, min_lr=0.000001)
clr = CyclicLR(base_lr=0.001, max_lr=0.006,step_size=2000., mode='triangular2')
# Please change the model name accordingly.
# mcp_save = ModelCheckpoint('model/aug_noiseNshift_2class2_np.h5', save_best_only=False, monitor='val_loss', mode='auto')
cnnhistory=model.fit(x_traincnn, y_train, batch_size=16, epochs=700,
                     validation_data=(x_testcnn, y_test), callbacks=[clr,es,ts])

# Plotting the Train Valid Loss Graph

plt.plot(cnnhistory.history['loss'])
plt.plot(cnnhistory.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save('C:\\Users\\danie\\Dropbox\\Classes\\programming\\Speech-Emotion-Analyzer-master\\Speech-Emotion-Analyzer-master\\saved_models\\saved_model.h5')
# model.save('C:\\Users\\noahd\\Desktop\\Speech-Emotion-Analyzer-master\\saved_model.h5')


# config = model.get_config()
# weights = model.get_weights()
model.save_weights('C:\\Users\\danie\\Dropbox\\Classes\\programming\\Speech-Emotion-Analyzer-master\\Speech-Emotion-Analyzer-master\\saved_weights.h5')
# model.save_weights('C:\\Users\\noahd\\Desktop\\Speech-Emotion-Analyzer-master\\saved_weights.h5')

# new_model = keras.Model.from_config(config)
# new_model.set_weights(weights)

# mew_model.save_weights('C:\\Users\\danie\\Dropbox\\Classes\\programming\\Speech-Emotion-Analyzer-master\\Speech-Emotion-Analyzer-master')
# model_name = 'Emotion_Voice_Detection_Model2.h5'
# save_dir = os.path.join(os.getcwd(), 'saved_models')


# Saving the model.json

import json
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)




