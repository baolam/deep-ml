import numpy as np

def calculate_correlation_matrix(X, Y=None):
	# Your code here
	X = np.array(X, dtype=float)
	if X.ndim == 1:
		X = X.reshape(-1, 1)

	if Y is None:
		Y = X
	else:
		Y = np.array(Y, dtype=float)
		if Y.ndim == 1:
			Y = Y.reshape(-1, 1)
	
	X_centered = X - np.mean(X, axis=0, keepdims=True)
	Y_centered = Y - np.mean(Y, axis=0, keepdims=True)

	std_X = np.std(X, axis=0, keepdims=True)
	std_Y = np.std(Y, axis=0, keepdims=True)

	covmatrix = (X_centered.T @ Y_centered) / X.shape[0]
	std = std_X.T @ std_Y

	with np.errstate(divide="ignore", invalid="ignore"):
		correlation_matrix = np.true_divide(covmatrix, std)
		correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
	
	return correlation_matrix