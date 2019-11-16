# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
# import os
# try:
# 	os.chdir(os.path.join(os.getcwd(), '..\\..\..\..\AppData\Local\Temp'))
# 	print(os.getcwd())
# except:
# 	pass
# %%
from IPython import get_ipython

# %% [markdown]
# ## Importing the required libraries

# %%
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix


# %%
from keras import regularizers


# %%
import os


# %%
mylist= os.listdir('RawData/')


# %%
type(mylist)


# %%
# print(mylist[1800])


# %%
print(mylist[400][6:-16])

# %% [markdown]
# ## Plotting the audio file's waveform and its spectrogram

# %%
data, sampling_rate = librosa.load('RawData/03-01-08-02-02-02-16.wav')


# %%
# get_ipython().run_line_magic('pylab', 'inline')
import os
import pandas as pd
import librosa
import glob 

plt.figure(figsize=(15, 5))
librosa.display.waveplot(data, sr=sampling_rate)


# %%
import matplotlib.pyplot as plt
import scipy.io.wavfile
import numpy as np
import sys


sr,x = scipy.io.wavfile.read('RawData/03-01-08-02-02-02-16.wav')

## Parameters: 10ms step, 30ms window
nstep = int(sr * 0.01)
nwin  = int(sr * 0.03)
nfft = nwin

window = np.hamming(nwin)

## will take windows x[n1:n2].  generate
## and loop over n2 such that all frames
## fit within the waveform
nn = range(nwin, len(x), nstep)

X = np.zeros( (len(nn), nfft//2) )

for i,n in enumerate(nn):
    xseg = x[n-nwin:n]
    z = np.fft.fft(window * xseg, nfft)
    X[i,:] = np.log(np.abs(z[:nfft//2]))

plt.imshow(X.T, interpolation='nearest',
    origin='lower',
    aspect='auto')

plt.show()

# %% [markdown]
# ## Setting the labels

# %%
feeling_list=[]
for item in mylist:
    if item[6:-16]=='02' and int(item[18:-4])%2==0:
        feeling_list.append('female_calm')
    elif item[6:-16]=='02' and int(item[18:-4])%2==1:
        feeling_list.append('male_calm')
    elif item[6:-16]=='03' and int(item[18:-4])%2==0:
        feeling_list.append('female_happy')
    elif item[6:-16]=='03' and int(item[18:-4])%2==1:
        feeling_list.append('male_happy')
    elif item[6:-16]=='04' and int(item[18:-4])%2==0:
        feeling_list.append('female_sad')
    elif item[6:-16]=='04' and int(item[18:-4])%2==1:
        feeling_list.append('male_sad')
    elif item[6:-16]=='05' and int(item[18:-4])%2==0:
        feeling_list.append('female_angry')
    elif item[6:-16]=='05' and int(item[18:-4])%2==1:
        feeling_list.append('male_angry')
    elif item[6:-16]=='06' and int(item[18:-4])%2==0:
        feeling_list.append('female_fearful')
    elif item[6:-16]=='06' and int(item[18:-4])%2==1:
        feeling_list.append('male_fearful')
    elif item[:1]=='a':
        feeling_list.append('male_angry')
    elif item[:1]=='f':
        feeling_list.append('male_fearful')
    elif item[:1]=='h':
        feeling_list.append('male_happy')
    #elif item[:1]=='n':
        #feeling_list.append('neutral')
    elif item[:2]=='sa':
        feeling_list.append('male_sad')


# %%
labels = pd.DataFrame(feeling_list)


# %%
labels[:10]

# %% [markdown]
# ## Getting the features of audio files using librosa

# %%
df = pd.DataFrame(columns=['feature'])
bookmark=0
for index,y in enumerate(mylist):
    if mylist[index][6:-16]!='01' and mylist[index][6:-16]!='07' and mylist[index][6:-16]!='08' and mylist[index][:2]!='su' and mylist[index][:1]!='n' and mylist[index][:1]!='d':
        X, sample_rate = librosa.load('RawData/'+y, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, 
                                            sr=sample_rate, 
                                            n_mfcc=13),
                        axis=0)
        feature = mfccs
        #[float(i) for i in feature]
        #feature1=feature[:135]
        df.loc[bookmark] = [feature]
        bookmark=bookmark+1        


# %%
df[:5]

# %% [markdown]
df3 = pd.DataFrame(df['feature'].values.tolist())
df3[:5]


# %%
newdf = pd.concat([df3,labels], axis=1)


# %%
rnewdf = newdf.rename(index=str, columns={"0": "label"})


# %%
rnewdf[:5]


# %%
from sklearn.utils import shuffle
rnewdf = shuffle(newdf)
rnewdf[:10]


# %%
rnewdf=rnewdf.fillna(0)

# %% [markdown]
# ## Dividing the data into test and train

# %%
newdf1 = np.random.rand(len(rnewdf)) < 0.8
train = rnewdf[newdf1]
test = rnewdf[~newdf1]


# %%
train[250:260]


# %%
trainfeatures = train.iloc[:, :-1]


# %%
trainlabel = train.iloc[:, -1:]


# %%
testfeatures = test.iloc[:, :-1]


# %%
testlabel = test.iloc[:, -1:]


# %%
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

X_train = np.array(trainfeatures)
y_train = np.array(trainlabel)
X_test = np.array(testfeatures)
y_test = np.array(testlabel)

lb = LabelEncoder()

y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))


# %%
y_train


# %%
X_train.shape

# %% [markdown]
# ## Changing dimension for CNN model

# %%

x_traincnn =np.expand_dims(X_train, axis=2)
x_testcnn= np.expand_dims(X_test, axis=2)


# %%
model = Sequential()

model.add(Conv1D(256, 5,padding='same',
                 input_shape=(216,1)))
model.add(Activation('relu'))
model.add(Conv1D(128, 5,padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
#model.add(Conv1D(128, 5,padding='same',))
#model.add(Activation('relu'))
#model.add(Conv1D(128, 5,padding='same',))
#model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))
opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)


# %%
model.summary()


# %%
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])

# %% [markdown]
# ### Removed the whole training part for avoiding unnecessary long epochs list

# %%
cnnhistory=model.fit(x_traincnn, y_train, batch_size=16, epochs=700, validation_data=(x_testcnn, y_test))


# %%
plt.plot(cnnhistory.history['loss'])
plt.plot(cnnhistory.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %% [markdown]
# ## Saving the model

# %%
model_name = 'Emotion_Voice_Detection_Model.h5'
save_dir = os.path.join(os.getcwd(), 'saved_models')
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)


# %%
import json
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# %% [markdown]
# ## Loading the model

# %%
# loading json and creating model
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("saved_models/Emotion_Voice_Detection_Model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
score = loaded_model.evaluate(x_testcnn, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

# %% [markdown]
# ## Predicting emotions on the test data

# %%
preds = loaded_model.predict(x_testcnn, 
                         batch_size=32, 
                         verbose=1)


# %%
print(preds)


# %%
preds1=preds.argmax(axis=1)


# %%
print(preds1)


# %%
abc = preds1.astype(int).flatten()


# %%
predictions = (lb.inverse_transform((abc)))


# %%
preddf = pd.DataFrame({'predictedvalues': predictions})
print(preddf[:10])


# %%
actual=y_test.argmax(axis=1)
abc123 = actual.astype(int).flatten()
actualvalues = (lb.inverse_transform((abc123)))


# %%
actualdf = pd.DataFrame({'actualvalues': actualvalues})
print(actualdf[:10])


# %%
finaldf = actualdf.join(preddf)

# %% [markdown]
# ## Actual v/s Predicted emotions

# %%
print(finaldf[170:180])


# %%
finaldf.groupby('actualvalues').count()


# %%
finaldf.groupby('predictedvalues').count()


# %%
finaldf.to_csv('Predictions.csv', index=False)

# %% [markdown]
# ## Live Demo
# %% [markdown]
# #### The file 'output10.wav' in the next cell is the file that was recorded live using the code in AudioRecoreder notebook found in the repository

# %%
data, sampling_rate = librosa.load('output10.wav')


# %%
# get_ipython().run_line_magic('pylab', 'inline')
import os
import pandas as pd
import librosa
import glob 

plt.figure(figsize=(15, 5))
librosa.display.waveplot(data, sr=sampling_rate)

plt.show()
# %%
#livedf= pd.DataFrame(columns=['feature'])
X, sample_rate = librosa.load('output10.wav', res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
sample_rate = np.array(sample_rate)
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
featurelive = mfccs
livedf2 = featurelive


# %%
livedf2= pd.DataFrame(data=livedf2)


# %%
livedf2 = livedf2.stack().to_frame().T


# %%
print(livedf2)


# %%
twodim= np.expand_dims(livedf2, axis=2)


# %%
livepreds = loaded_model.predict(twodim, 
                         batch_size=32, 
                         verbose=1)


# %%
print(livepreds)


# %%
livepreds1=livepreds.argmax(axis=1)


# %%
liveabc = livepreds1.astype(int).flatten()


# %%
livepredictions = (lb.inverse_transform((liveabc)))
print(livepredictions)


# %%


