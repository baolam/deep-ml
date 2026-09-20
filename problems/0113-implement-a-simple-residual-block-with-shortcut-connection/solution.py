import numpy as np

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
	# Your code here
	h1 = np.dot(w1, x)
	a1 = np.maximum(0, h1)
	h2 = np.dot(a1, w2)
	a2 = np.maximum(0, h2 + x)
	return a2