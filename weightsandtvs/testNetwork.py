# Base CNN imports
import pandas as panda
import os
import tensorflow as tf
import numpy as np
from numpy import mean, sqrt, square
import random
import matplotlib
matplotlib.use('Agg')
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

paths = ["tvs", "weights"]
sequence_length = 30
	
for root, dirs, files in chain.from_iterable(os.walk(path) for path in paths):
	for file in sorted(files):
		if (file.endswith(".npy")):
			path = os.path.join(root, file)
			if (root == "tvs") :
				print(path)
				trackList.append(path)
			elif (root == "weights") :
				print(path)
				mfccsList.append(path)
						
input_first = (Input(shape=(sequence_length, 10), name='mfccInput'))
input_next = CuDNNLSTM(10, input_shape=(sequence_length, 10), return_sequences=True, name="LSTM_aggregation2")(input_first)
input_next = CuDNNLSTM(10, input_shape=(sequence_length, 10), return_sequences=False, name="LSTM_aggregation3")(input_next)
predictions = (Dense(6, activation="linear", name="predictions", input_shape=(10,)))(input_next)

# Use the Adam method for training the network.
# We want to find the best learning-rate for the Adam method.
optimizer = Adam(lr=1e-4)
model = Model(inputs=[input_first], outputs=predictions)   
homepath = os.environ["HOME"]
path_best_model = '{}/10_17_tvweights_LSTM_Regression_Model_SAIL_SPEECH.keras'.format(homepath)
# model = Model(inputs=[input_first], outputs=predictions)    
model = load_model(path_best_model)
# In Keras we need to compile the model so it can be trained.
model.compile(optimizer=optimizer,
			  loss='mean_squared_error',
			  metrics=['mse'])

			  
			  
mfcc = mfccsList[0]
track = trackList[0]
mfccs = (np.load(mfcc))
sixDistances = np.load(track)

mfcc_array = []
sixDistances_array = []

path_stdscaler = '{}/10_17_tvweight_StandardScaler.pkl'.format(homepath)
path_mmscaler = '{}/10_17_tvweight_MinMaxScaler.pkl'.format(homepath)
fitter = joblib.load(path_stdscaler) 
scaler = joblib.load(path_mmscaler) 

mfccs = fitter.fit_transform(mfccs)
#sixDistances = scaler.fit_transform(sixDistances)

for i in range(30, len(sixDistances)-1):
	sample_mfcc = mfccs[i-sequence_length:i]
	sample_sixDistances = sixDistances[i]
	mfcc_array.append(sample_mfcc)
	sixDistances_array.append(sample_sixDistances)

real_mfcc_array = np.asarray(mfcc_array, dtype=np.float32)
real_sixDistances_array = np.asarray(sixDistances_array, dtype=np.float32)

index = np.shape(real_sixDistances_array)[0]

# Make predictions, scale them, and convert to mm along with ground truth
score = model.predict(x=real_mfcc_array[index-30000:index])
score = scaler.inverse_transform(score)

sumErrors = np.zeros(shape=(1, 6));
for i in range(len(score)):
	for j in range(6):
		sumErrors[0][j] += square(score[i][j] - real_sixDistances_array[i][j])

for k in range(6):
	sumErrors[0][k] /= len(score)
	sumErrors[0][k] = sqrt(sumErrors[0][k])
	print('error {}: {}'.format(k, sumErrors[0][k]))
	
#print(score[0][29])
plt.plot(score[:,0])
plt.plot(real_sixDistances_array[:,0])
plt.legend(['Model score', 'Ground Truth'], loc='upper left')
plt.savefig("tvsandweights.png")
