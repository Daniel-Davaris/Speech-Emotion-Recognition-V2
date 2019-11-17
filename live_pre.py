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
from sklearn.metrics import confusion_matrix
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
# from tensorflow.keras import backend

# loading json and creating model
from keras.models import model_from_json
from keras import backend as K


input_duration=3

tmp1 = pd.read_pickle("./dummy.pkl")
tmp3 = pd.read_pickle("./dummy.pkl")
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




json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)



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

# load_weights(fpath)

# weights = model.get_weights('C:\\Users\\danie\\Dropbox\\Classes\\programming\\Speech-Emotion-Analyzer-master\\Speech-Emotion-Analyzer-master\\.h5')

# new_model = keras.models.load_model('C:\\Users\\danie\\Dropbox\\Classes\\programming\\Speech-Emotion-Analyzer-master\\Speech-Emotion-Analyzer-master\\saved_models\\saved_model.h5')
# new_model.load_weights('C:\\Users\\danie\\Dropbox\\Classes\\programming\\Speech-Emotion-Analyzer-master\\Speech-Emotion-Analyzer-master\\.h5')


# new_model = tf.keras.models.load_model('C:\\Users\\noahd\\desktop\\Speech-Emotion-Analyzer-master\\saved_models\\saved_model.h5', custom_objects={'fscore': fscore})
new_model = tf.keras.models.load_model('C:\\Users\\danie\\dropbox\\classes\\programming\\final product\\saved_models\\saved_model.h5', custom_objects={'fscore': fscore})

new_model.summary()



test_valid_lb = np.array(data3_df.label)
lb = LabelEncoder()
test_valid_lb = np_utils.to_categorical(lb.fit_transform(test_valid_lb))




data, sampling_rate = librosa.load('output10.wav')
plt.figure(figsize=(15, 5))
librosa.display.waveplot(data, sr=sampling_rate)

plt.title('Mel power spectrogram ')

plt.tight_layout()
plt.show()



X, sample_rate = librosa.load('output10.wav', res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
sample_rate = np.array(sample_rate)
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
featurelive = mfccs
livedf2 = featurelive
print(livedf2.shape)
livedf2= pd.DataFrame(data=livedf2)

livedf2 = livedf2.stack().to_frame().T

print(livedf2)
print(livedf2.shape)





def plot_time_series(data):
    """
    Plot the Audio Frequency.
    """
    fig = plt.figure(figsize=(14, 8))
    plt.title('Raw wave ')
    plt.ylabel('Amplitude')
    plt.plot(np.linspace(0, 1, len(data)), data)
    plt.show()


def noise(data):
    """
    Adding White Noise.
    """
    # you can take any distribution from https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html
    noise_amp = 0.005*np.random.uniform()*np.amax(data)
    data = data.astype('float64') + noise_amp * np.random.normal(size=data.shape[0])
    return data
    
def shift(data):
    """
    Random Shifting.
    """
    s_range = int(np.random.uniform(low=-5, high = 5)*500)
    return np.roll(data, s_range)
    
def stretch(data, rate=0.8):
    """
    Streching the Sound.
    """
    data = librosa.effects.time_stretch(data, rate)
    return data

def pitch(data, sample_rate):
    """
    Pitch Tuning.
    """
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change =  pitch_pm * 2*(np.random.uniform())   
    data = librosa.effects.pitch_shift(data.astype('float64'), 
                                      sample_rate, n_steps=pitch_change, 
                                      bins_per_octave=bins_per_octave)
    return data
    
def dyn_change(data):
    """
    Random Value Change.
    """
    dyn_change = np.random.uniform(low=1.5,high=3)
    return (data * dyn_change)
    
def speedNpitch(data):
    """
    peed and Pitch Tuning.
    """
    # you can change low and high here
    length_change = np.random.uniform(low=0.8, high = 1)
    speed_fac = 1.0  / length_change
    tmp = np.interp(np.arange(0,len(data),speed_fac),np.arange(0,len(data)),data)
    minlen = min(data.shape[0], tmp.shape[0])
    data *= 0
    data[0:minlen] = tmp[0:minlen]
    return data








livedf2 = np.resize(livedf2,(1,259))    

twodim= np.expand_dims(livedf2, axis=2)

livepreds = new_model.predict(twodim, 
                         batch_size=32, 
                         verbose=1)

print(livepreds)


livepreds1=livepreds.argmax(axis=1)

liveabc = livepreds1.astype(int).flatten()

livepredictions = (lb.inverse_transform((liveabc)))
print(livepredictions)
# evaluate loaded model on test data
opt = keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
new_model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
score = new_model.evaluate(x_testcnn, y_test, verbose=0)
print("%s: %.2f%%" % (new_model.metrics_names[1], score[1]*100))


