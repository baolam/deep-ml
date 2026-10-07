import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	# Your code here
	def _sigmoid(z):
		return 1 / (1 + np.exp(-z))

	updated_weights = initial_weights.copy().astype(float)
	updated_bias = initial_bias

	def _predict():
		return _sigmoid(np.dot(features, updated_weights) + updated_bias)
	
	m = features.shape[0]
	mse_values = []

	for _ in range(epochs):
		pred = _predict()
		err = pred - labels

		mse_values.append(
			round(float((err ** 2).mean()), 4)
		)

		dz = (2 / m) * err * pred * (1 - pred)
		dw = np.dot(features.T, dz)
		db = np.sum(dz)

		updated_weights -= learning_rate * dw
		updated_bias -= learning_rate * db
	
	updated_weights = np.round(updated_weights, 4).tolist()
	updated_bias = round(float(updated_bias), 4)

	return updated_weights, updated_bias, mse_values