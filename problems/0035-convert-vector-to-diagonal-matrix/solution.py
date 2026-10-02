import numpy as np

def make_diagonal(x):
	# Your code here
	x = np.array(x, dtype=np.float32)
	return np.diag(x)