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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from itertools import chain

mfccsList = []
trackList = []

paths = ["csvs", "mfccs"]
sequence_length = 30
	
for root, dirs, files in chain.from_iterable(os.walk(path) for path in paths):
	for file in sorted(files):
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
input_next = CuDNNLSTM(13, input_shape=(sequence_length, 13), return_sequences=False, name="LSTM_aggregation2")(input_next)
predictions = (Dense(6, activation="linear", name="predictions", input_shape=(13,)))(input_next)

# Use the Adam method for training the network.
# We want to find the best learning-rate for the Adam method.
optimizer = Adam(lr=1e-3)
model = Model(inputs=[input_first], outputs=predictions)    
# model = load_model('09_12_LSTM_Regression_Model')
# In Keras we need to compile the model so it can be trained.
model.compile(optimizer=optimizer,
			  loss='mean_squared_error',
			  metrics=['mse'])
			  
pmodel = multi_gpu_model(model, gpus=4)
pmodel.compile(optimizer=optimizer,
			  loss='mean_squared_error',
			  metrics=['mse'])
			  
scaler = MinMaxScaler(feature_range=(0.0, 1.0))
fitter = StandardScaler()			  
for mfcc, track in zip(mfccsList, trackList):
	mfccs = np.transpose(np.load(mfcc))
	sixDistances = np.load(track)
	fitter.partial_fit(mfccs)
	scaler.partial_fit(sixDistances)

joblib.dump(fitter, 'StandardScaler.pkl') 
joblib.dump(scaler, 'MinMaxScaler.pkl') 

counter = 0	
for mfcc, track in zip(mfccsList, trackList):
	print(mfcc)
	print(track)
	mfccs = np.transpose(np.load(mfcc))
	sixDistances = np.load(track)
	print(len(mfccs))
	print(sixDistances.shape)

	mfcc_array = []
	sixDistances_array = []
	
	mfccs = fitter.fit_transform(mfccs)
	sixDistances = scaler.fit_transform(sixDistances)
	
	for i in range(30, len(sixDistances)-1):
		sample_mfcc = mfccs[i-sequence_length:i]
		sample_sixDistances = sixDistances[i]
		mfcc_array.append(sample_mfcc)
		print(sample_mfcc)
		sixDistances_array.append(sample_sixDistances)

	real_mfcc_array = np.asarray(mfcc_array, dtype=np.float32)
	real_sixDistances_array = np.asarray(sixDistances_array, dtype=np.float32)

	if (counter < 9):
		index = np.shape(real_sixDistances_array)[0]
		crossValidation = index-500
		testingSet = index-300

		validation_data = (real_mfcc_array[crossValidation:testingSet], real_sixDistances_array[crossValidation:testingSet])
		print(real_mfcc_array[crossValidation:testingSet])
		history = pmodel.fit(x=real_mfcc_array[0:crossValidation],
							y=real_sixDistances_array[0:crossValidation],
							epochs=500,
							batch_size=512,
							validation_data=validation_data)
							
		model.save("10_09_LSTM_Regression_Model_retest.keras")

	else:
		score = model.predict(x=real_mfcc_array)
		scorepath = "score_" + track
		np.save(scorepath, score)
		
	counter += 1