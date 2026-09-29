import numpy as np

def feature_scaling(data):
	mean = data.mean(axis=0, keepdims=True)
	std = data.std(axis=0, keepdims=True)
	z = np.round((data - mean) / std, 4)
	_min = data.min(axis=0, keepdims=True)
	_max = data.max(axis=0, keepdims=True)
	mm = np.round((data - _min) / (_max - _min), 4)
	return z, mm