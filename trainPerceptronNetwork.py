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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from itertools import chain

import horovod.keras as hvd

mfccsList = []
trackList = []

for dataset in sorted(os.listdir("csvs")):
	data_dir = 'csvs/{}'.format(dataset)
	data_files = os.listdir(data_dir)
	for file in sorted(data_files):
		if (file.endswith(".npy")):
			trackList.append(os.path.join(data_dir, file))

for dataset in sorted(os.listdir("mfccs")):
	data_dir = 'mfccs/{}'.format(dataset)
	data_files = os.listdir(data_dir)
	for file in sorted(data_files):
		if (file.endswith(".npy")):
			mfccsList.append(os.path.join(data_dir, file))
			
input_first = (Input(shape=(13,), name='mfccInput'))
predictions = (Dense(6, activation="relu", name="predictions", input_shape=(13,)))(input_first)

# Use the Adam method for training the network.
# We want to find the best learning-rate for the Adam method.
optimizer = Adam(lr=1e-3)
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
			  
# Scale mfccs mean 0 and standard deviation of 1
fitter = StandardScaler()

# Scale six constriction values with range 0, 1 since they are all positive
scaler = MinMaxScaler(feature_range=(0.0, 1.0))
		
counter = 0			
for mfcc, track in zip(mfccsList, trackList):
	mfccs = np.transpose(np.load(mfcc))
	sixDistances = np.load(track)
	fitter.partial_fit(mfccs)
	scaler.partial_fit(sixDistances)
	counter += 1

homepath = os.environ["HOME"]
path_stdscaler = '{}/1_18_Perceptron_StandardScaler.pkl'.format(homepath)
path_mmscaler = '{}/1_18_Perceptron_MinMaxScaler.pkl'.format(homepath)
joblib.dump(fitter, path_stdscaler) 
joblib.dump(scaler, path_mmscaler) 

fileIndex = 0
for mfcc, track in zip(mfccsList, trackList):
	
	if (fileIndex < counter - 3):

		real_mfcc_array = np.transpose(np.load(mfcc))
		real_sixDistances_array = np.load(track)

		real_mfcc_array = fitter.fit_transform(real_mfcc_array)
		real_sixDistances_array = scaler.fit_transform(real_sixDistances_array)
		
		index = np.shape(real_sixDistances_array)[0]
		crossValidation = index-500
		testingSet = index

		validation_data = (real_mfcc_array[crossValidation:testingSet], real_sixDistances_array[crossValidation:testingSet])

		history = model.fit(x=real_mfcc_array[0:crossValidation],
							y=real_sixDistances_array[0:crossValidation],
							epochs=25000,
							batch_size=512,
							validation_data=validation_data)
		
		path_best_model = '{}/1_18_Perceptron_Regression_Model_SAIL_SPEECH_1M.keras'.format(homepath)
		if hvd.rank() == 0:
			model.save(path_best_model)

	fileIndex += 1