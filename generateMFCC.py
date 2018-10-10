import librosa
import numpy as np
import os

path = r"B:\ownCloud\usc_vtsf_m1\2D\wav"

for root, dirs, files in os.walk(path):
	for filename in sorted(files):
		p=os.path.join(root,filename)
		y, fs = librosa.load(p)
		mfccs = librosa.feature.mfcc(y=y, sr=fs, n_mfcc=13, hop_length=int(0.01*fs), n_fft=int(0.025*fs))
		print(mfccs.shape)
		prefix_fname = filename.replace('.', '_')
		imagePath = '{}_mfccs'.format(prefix_fname)
		np.save(imagePath, mfccs)