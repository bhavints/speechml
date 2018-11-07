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

import horovod.keras as hvd

mfccsList = []
trackList = []

sequence_length = 30
	
for dataset in os.listdir("csvs"):
	data_dir = 'csvs/{}'.format(dataset)
	data_files = os.listdir(data_dir)
	for file in sorted(data_files):
		if (file.endswith(".npy")):
			trackList.append(os.path.join(data_dir, file))

for dataset in os.listdir("mfccs"):
	data_dir = 'mfccs/{}'.format(dataset)
	data_files = os.listdir(data_dir)
	for file in sorted(data_files):
		if (file.endswith(".npy")):
			mfccsList.append(os.path.join(data_dir, file))
			
input_first = (Input(shape=(sequence_length, 13), name='mfccInput'))
input_next = CuDNNLSTM(13, input_shape=(sequence_length, 13), return_sequences=True, name="LSTM_aggregation")(input_first)
input_next = CuDNNLSTM(13, input_shape=(sequence_length, 13), return_sequences=True, name="LSTM_aggregation2")(input_next)
input_next = CuDNNLSTM(13, input_shape=(sequence_length, 13), return_sequences=False, name="LSTM_aggregation3")(input_next)
predictions = (Dense(6, activation="linear", name="predictions", input_shape=(13,)))(input_next)

# Use the Adam method for training the network.
# We want to find the best learning-rate for the Adam method.
optimizer = Adam(lr=1e-4)
model = Model(inputs=[input_first], outputs=predictions)    
# model = load_model('09_12_LSTM_Regression_Model')
# In Keras we need to compile the model so it can be trained.
#model.compile(optimizer=optimizer,
#			  loss='mean_squared_error',
#			  metrics=['mse'])


# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))

optimizer = hvd.DistributedOptimizer(optimizer)

model.compile(optimizer=optimizer,
	loss='mean_squared_error',
	metrics=['mse'])	

callbacks = [
	# Horovod: broadcast initial variable states from rank 0 to all other processes.
	# This is necessary to ensure consistent initialization of all workers when
	# training is started with random weights or restored from a checkpoint.
	hvd.callbacks.BroadcastGlobalVariablesCallback(0)
]
	
#pmodel = multi_gpu_model(model, gpus=4)
#pmodel.compile(optimizer=optimizer,
#			  loss='mean_squared_error',
#			  metrics=['mse'])
			  
scaler = MinMaxScaler(feature_range=(0.0, 1.0))
fitter = StandardScaler()
			  
for mfcc, track in zip(mfccsList, trackList):
	mfccs = np.transpose(np.load(mfcc))
	sixDistances = np.load(track)
	fitter.partial_fit(mfccs)
	scaler.partial_fit(sixDistances)

homepath = os.environ["HOME"]
path_stdscaler = '{}/11_7_StandardScaler.pkl'.format(homepath)
path_mmscaler = '{}/11_7_MinMaxScaler.pkl'.format(homepath)
joblib.dump(fitter, path_stdscaler) 
joblib.dump(scaler, path_mmscaler) 

for mfcc, track in zip(mfccsList, trackList):
	print(mfcc)
	print(track)
	mfccs = np.transpose(np.load(mfcc))
	real_sixDistances_array = np.load(track)
	print(len(mfccs))
	print(real_sixDistances_array.shape)

	mfcc_array = []
	sixDistances = []
	
	mfccs = fitter.fit_transform(mfccs)
	real_sixDistances_array = scaler.fit_transform(real_sixDistances_array)
	
	for i in range(len(real_sixDistances_array)):
		if (i > sequence_length-1):
			
			sample_mfcc = mfccs[i-sequence_length:i]
			mfcc_array.append(sample_mfcc)
			
			sixDistances.append(real_sixDistances_array[i])

	
	real_mfcc_array = np.asarray(mfcc_array, dtype=np.float32)
	real_sixDistances_array = np.asarray(sixDistances, dtype=np.float32)
	
	index = np.shape(real_sixDistances_array)[0]
	crossValidation = index-500
	testingSet = index

	validation_data = (real_mfcc_array[crossValidation:testingSet], real_sixDistances_array[crossValidation:testingSet])
	print(real_mfcc_array[crossValidation:testingSet])
	history = model.fit(x=real_mfcc_array[0:crossValidation],
						y=real_sixDistances_array[0:crossValidation],
						epochs=5000,
						batch_size=2048,
						validation_data=validation_data)
	
	path_best_model = '{}/11_7_LSTM_Regression_Model_SAIL_SPEECH_1M.keras'.format(homepath)
	if hvd.rank() == 0:
		model.save(path_best_model)
