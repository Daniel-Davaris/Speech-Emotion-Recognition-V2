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
from keras import regularizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.callbacks import  History, ReduceLROnPlateau, CSVLogger
from keras.models import Model, Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
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



# Data location

direct = os.listdir('data/')
 

# DataFrame object for files
data_array = pd.DataFrame(columns=['path', 'source', 'actor', 'gender','intensity', 'statement', 'repetition', 'emotion'])
count = 0
for i in direct:
    audio_files = os.listdir('data/' + i)
    for f in audio_files:
        file_split = f.split('.')[0].split('-')
        path = 'data/' + i + '/' + f
        src = int(file_split[1])
        actor = int(file_split[-1])
        emotion = int(file_split[2])
        if int(actor)%2 == 0:
            gender = "female"
        else:
            gender = "male"
        if file_split[3] == '01':
            intensity = 0
        else:
            intensity = 1
        if file_split[4] == '01':
            statement = 0
        else:
            statement = 1
        if file_split[5] == '01':
            repeat = 0
        else:
            repeat = 1

        data_array.loc[count] = [path, src, actor, gender, intensity, statement, repeat, emotion]
        count += 1

# defining the loabels

labels= []
for i in range(len(data_array)):
    if data_array.emotion[i] == 2: 
        lb = "_positive"
    elif data_array.emotion[i] == 3: 
        lb = "_positive"
    elif data_array.emotion[i] == 4:
        lb = "_negative"
    elif data_array.emotion[i] == 5: 
        lb = "_negative"
    elif data_array.emotion[i] == 6: 
        lb = "_negative"
    else:
        lb = "_none" 
    labels.append(data_array.gender[i] + lb)
    

data_array['label'] = labels


#Dividing up the data
df_obj = data_array.copy()
df_obj = df_obj[df_obj.label != "male_none"]
df_obj = df_obj[df_obj.label != "female_none"].reset_index(drop=True)
df_obj = df_obj[df_obj.label != "female_neutral"]
df_obj = df_obj[df_obj.label != "female_happy"]
df_obj = df_obj[df_obj.label != "female_angry"]
df_obj = df_obj[df_obj.label != "female_sad"]
df_obj = df_obj[df_obj.label != "female_fearful"]
df_obj = df_obj[df_obj.label != "female_calm"]
df_obj = df_obj[df_obj.label != "female_positive"]
df_obj = df_obj[df_obj.label != "female_negative"].reset_index(drop=True)
tmp1 = df_obj[df_obj.actor == 21]
tmp2 = df_obj[df_obj.actor == 22]
tmp3 = df_obj[df_obj.actor == 23]
tmp4 = df_obj[df_obj.actor == 24]
data3_df = pd.concat([tmp1, tmp3],ignore_index=True).reset_index(drop=True)
df_obj = df_obj[df_obj.actor != 21]
df_obj = df_obj[df_obj.actor != 22]
df_obj = df_obj[df_obj.actor != 23].reset_index(drop=True)
df_obj = df_obj[df_obj.actor != 24].reset_index(drop=True)
data3_df = np.array(data3_df)
original_df = tmp1
original_df2 = tmp3


os.remove("./dummy.pkl")
original_df.to_pickle("./dummy.pkl")

os.remove("./dummy2.pkl")
original_df2.to_pickle("./dummy2.pkl")





# actual feature extration
data = pd.DataFrame(columns=['feature'])
for i in tqdm(range(len(df_obj))):
    X, sample_rate = librosa.load(df_obj.path[i], res_type='kaiser_fast',duration=3,sr=22050*2,offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
    feature = mfccs
    data.loc[i] = [feature]


df3 = pd.DataFrame(data['feature'].values.tolist())
labels = df_obj.label
newdf = pd.concat([df3,labels], axis=1)
rnewdf = newdf.rename(index=str, columns={"0": "label"})
rnewdf = rnewdf.fillna(0)

# White Noise function
def noise(data):
    
    noise_amp = 0.005*np.random.uniform()*np.amax(data)
    data = data.astype('float64') + noise_amp * np.random.normal(size=data.shape[0])
    return data
#random shifting function
def shift(data):
    s_range = int(np.random.uniform(low=-5, high = 5)*500)
    return np.roll(data, s_range)
#streching function
def stretch(data, rate=0.8):
    data = librosa.effects.time_stretch(data, rate)
    return data
# pitch tuning
def pitch(data, sample_rate):
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change =  pitch_pm * 2*(np.random.uniform())   
    data = librosa.effects.pitch_shift(data.astype('float64'), sample_rate, n_steps=pitch_change, bins_per_octave=bins_per_octave)
    return data
#random value change function 
def dyn_change(data):
    dyn_change = np.random.uniform(low=1.5,high=3)
    return (data * dyn_change)
#Tuning function
def speedNpitch(data):
    length_change = np.random.uniform(low=0.8, high = 1)
    speed_fac = 1.0  / length_change
    tmp = np.interp(np.arange(0,len(data),speed_fac),np.arange(0,len(data)),data)
    minlen = min(data.shape[0], tmp.shape[0])
    data *= 0
    data[0:minlen] = tmp[0:minlen]
    return data




# data augmentation 

syn_data1 = pd.DataFrame(columns=['feature', 'label'])
for i in tqdm(range(len(df_obj))):
    X, sample_rate = librosa.load(df_obj.path[i], res_type='kaiser_fast',duration=3,sr=22050*2,offset=0.5)
    if df_obj.label[i]:

        X = noise(X)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
        feature = mfccs
        a = random.uniform(0, 1)
        syn_data1.loc[i] = [feature, df_obj.label[i]]

syn_data2 = pd.DataFrame(columns=['feature', 'label'])
for i in tqdm(range(len(df_obj))):
    X, sample_rate = librosa.load(df_obj.path[i], res_type='kaiser_fast',duration=3,sr=22050*2,offset=0.5)
    if df_obj.label[i]:
        X = pitch(X, sample_rate)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
        feature = mfccs
        a = random.uniform(0, 1)
        syn_data2.loc[i] = [feature, df_obj.label[i]]



syn_data1 = syn_data1.reset_index(drop=True)
syn_data2 = syn_data2.reset_index(drop=True)
df4 = pd.DataFrame(syn_data1['feature'].values.tolist())
labels4 = syn_data1.label
syndf1 = pd.concat([df4,labels4], axis=1)
syndf1 = syndf1.rename(index=str, columns={"0": "label"})
syndf1 = syndf1.fillna(0)
df4 = pd.DataFrame(syn_data2['feature'].values.tolist())
labels4 = syn_data2.label
syndf2 = pd.concat([df4,labels4], axis=1)
syndf2 = syndf2.rename(index=str, columns={"0": "label"})
syndf2 = syndf2.fillna(0)
combined_df = pd.concat([rnewdf, syndf1, syndf2], ignore_index=True)
combined_df = combined_df.fillna(0)
X = combined_df.drop(['label'], axis=1)
y = combined_df.label
xxx = StratifiedShuffleSplit(1, test_size=0.2, random_state=12)
for train_index, test_index in xxx.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
# test and train split 
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))
x_traincnn = np.expand_dims(X_train, axis=2)
x_testcnn = np.expand_dims(X_test, axis=2)





#exporting the data to the model
os.remove('data.h5')
h5f = h5py.File('data.h5', 'w-')
h5f.create_dataset('one', data=x_traincnn)     
h5f.create_dataset('two', data=x_testcnn)    
h5f.create_dataset('three', data=y_test) 
h5f.create_dataset('four', data=y_train) 
h5f.create_dataset('six', data=X_test) 
h5f.create_dataset('seven', data=X_train) 
h5f.close()




