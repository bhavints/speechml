# Base CNN imports
import pandas as panda
import os
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from numbers import Number
from sklearn.externals import joblib 

# For parameter tuning
import math

# Keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input, concatenate, BatchNormalization
from tensorflow.python.keras.layers import Reshape, MaxPooling2D, Lambda, TimeDistributed
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, CuDNNLSTM, ConvLSTM2D 
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import save_model, load_model, Model
from tensorflow.python.keras.utils import multi_gpu_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from itertools import chain

mfccsList = []
trackList = []

paths = ["csvs", "mfccs"]
sequence_length = 30
	
for root, dirs, files in chain.from_iterable(os.walk(path) for path in paths):
	for file in files:
		if (file.endswith(".npy")):
			path = os.path.join(root, file)
			if (root == "csvs") :
				print(path)
				trackList.append(path)
			elif (root == "mfccs") :
				print(path)
				mfccsList.append(path)
						
input_first = (Input(shape=(sequence_length, 13), name='mfccInput'))
input_next = CuDNNLSTM(13, input_shape=(sequence_length, 13), return_sequences=True, name="LSTM_aggregation")(input_first)
input_next = CuDNNLSTM(6, input_shape=(sequence_length, 13), return_sequences=True, name="LSTM_aggregation2")(input_next)
predictions = TimeDistributed(Dense(6, activation="linear", name="predictions", input_shape=(sequence_length,6)))(input_next)

# Use the Adam method for training the network.
# We want to find the best learning-rate for the Adam method.
optimizer = Adam(lr=1e-3)
# model = Model(inputs=[input_first], outputs=predictions)    
model = load_model("10_09_LSTM_Regression_Model_SAIL_SPEECH.keras")
# In Keras we need to compile the model so it can be trained.
model.compile(optimizer=optimizer,
			  loss='mean_squared_error',
			  metrics=['mse'])

			  
			  
mfcc = mfccsList[10]
track = trackList[10]
print(mfcc)
print(track)
mfccs = np.transpose(np.load(mfcc))
sixDistances = np.load(track)
print(len(mfccs))
print(sixDistances.shape)

mfcc_array = []
sixDistances_array = []

fitter = joblib.load('StandardScaler.pkl') 
scaler = joblib.load('MinMaxScaler.pkl') 

mfccs = fitter.fit_transform(mfccs)
sixDistances = scaler.fit_transform(sixDistances)

for i in range(30, len(sixDistances)-1):
	sample_mfcc = mfccs[i-sequence_length:i]
	sample_sixDistances = sixDistances[i]
	mfcc_array.append(sample_mfcc)
	sixDistances_array.append(sample_sixDistances)

real_mfcc_array = np.asarray(mfcc_array, dtype=np.float32)
real_sixDistances_array = np.asarray(sixDistances_array, dtype=np.float32)

index = np.shape(real_sixDistances_array)[0]

score = model.predict(x=real_mfcc_array)
#print(score[0][29])
plt.plot(score[:,0])
plt.plot(real_sixDistances_array[:,0])
plt.legend(['Model score', 'Ground Truth'], loc='upper left')
plt.show()
