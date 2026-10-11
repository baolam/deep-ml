import numpy as np

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
	"""
	Implements binary classification prediction using Logistic Regression.

	Args:
		X: Input feature matrix (shape: N x D)
		weights: Model weights (shape: D)
		bias: Model bias

	Returns:
		Binary predictions (0 or 1)
	"""
	# Your code here
	def _sigmoid(z):
		return 1 / (1 + np.exp(-z))
	
	h = X @ weights + bias
	a = _sigmoid(h)

	return (a >= 0.5).astype(int)