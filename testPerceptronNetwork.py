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
		
# Use the Adam method for training the network.
# We want to find the best learning-rate for the Adam method.
optimizer = Adam(lr=1e-4)
  
homepath = os.environ["HOME"]
path_stdscaler = '{}/11_28_Perceptron_StandardScaler.pkl'.format(homepath)
path_mmscaler = '{}/11_28_Perceptron_MinMaxScaler.pkl'.format(homepath)
fitter = joblib.load(path_stdscaler) 
scaler = joblib.load(path_mmscaler)

path_best_model = '{}/11_28_PERCEPTRON_Regression_Model_SAIL_SPEECH_1M.keras'.format(homepath)  
model = load_model(path_best_model)

model.compile(optimizer=optimizer,
	loss='mean_squared_error',
	metrics=['mse'])	
		  		
counter = 0			
for mfcc, track in zip(mfccsList, trackList):
	mfccs = np.transpose(np.load(mfcc))
	sixDistances = np.load(track)
	fitter.partial_fit(mfccs)
	scaler.partial_fit(sixDistances)
	counter += 1

fileIndex = 0
for mfcc, track in zip(mfccsList, trackList):
	
	if (fileIndex >= counter - 3):

		real_mfcc_array = np.transpose(np.load(mfcc))
		real_sixDistances_array = np.load(track)

		real_mfcc_array = fitter.fit_transform(real_mfcc_array)
		real_sixDistances_array = scaler.fit_transform(real_sixDistances_array)
		
		index = np.shape(real_sixDistances_array)[0]

		# Make predictions, scale them, and convert to mm along with ground truth
		score = model.predict(x=real_mfcc_array)
		score = scaler.inverse_transform(score)
		score *= 2.4
		real_sixDistances_array *= 2.4

		sumErrors = np.zeros(shape=(1, 6));
		for i in range(len(score)):
			for j in range(6):
				sumErrors[0][j] += square(score[i][j] - real_sixDistances_array[i][j])

		print('ERROR FOR {}'.format(fileIndex), file=open("perceptronerrors.txt", "a"))
		for k in range(6):
			sumErrors[0][k] /= len(score)
			sumErrors[0][k] = sqrt(sumErrors[0][k])
			print('error {}: {}'.format(k, sumErrors[0][k]), file=open("errors.txt", "a"))
		
		for i in range(1000, len(score), 1000) :
			#print(score[0][29])
			plt.plot(score[i-1000:i])
			plt.plot(real_sixDistances_array[i-1000:i])
			plt.legend(['Model score', 'Ground Truth'], loc='upper left')
			plt.savefig('perceptron_results_track_{}_of_10_normalized_{}.png'.format(fileIndex, i))

	fileIndex += 1