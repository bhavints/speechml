import numpy as np
from scipy import stats

sixDistances = np.load("mfccs/cg/usc_vtsf_m001_rt_bVt_r1_wav_mfccs.npy")

for i in range(6):
	singleDistance = sixDistances[:, i]
	singleDistance = singleDistance.transpose()
	print(singleDistance.shape)
	print(stats.describe(singleDistance))