import numpy as np

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	# Your code here
	X = np.array(vectors)

	n_features, n_samples = X.shape

	mean = np.mean(X, axis=1, keepdims=True)
	X_centered = X - mean

	cov = (X_centered @ X_centered.T) / (n_samples - 1)
	return cov.tolist()