# Base CNN imports
import pandas as panda
import numpy as np
import os

for root, dirs, files in os.walk("CSVs"):
	for file in files:
		if (file.endswith(".csv")):
			path = os.path.join(root, file)
			prefix_fname = os.path.splitext(path)[0]+'.npy'
			data = panda.read_csv(path, header=None, dtype='float32')
			print(data[0])
			np.save(prefix_fname, data)

