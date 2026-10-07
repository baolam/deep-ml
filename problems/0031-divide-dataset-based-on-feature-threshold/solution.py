import numpy as np

def divide_on_feature(X, feature_i, threshold):
	# Your code here
	X = np.array(X)
	mask = X[:, feature_i] >= threshold

	X1 = X[mask]
	X2 = X[~mask]

	return [X1, X2]