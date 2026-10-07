import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	# Your code here
	weights = initial_weights.copy().astype(float)
	bias = initial_bias
	mse_values = []

	m = features.shape[0]

	def _sigmoid(z):
		return 1 / (1 + np.exp(-z))

	for _ in range(epochs):
		pred = np.dot(features, weights) + bias
		pred = _sigmoid(pred)
		err = pred - labels
		mse = np.mean(err ** 2)

		mse_values.append(round(float(mse), 4))

		dz = (2 / m) * err * pred * (1 - pred)
		dw = np.dot(features.T, dz)
		db = np.sum(dz)

		weights -= learning_rate * dw
		bias -= learning_rate * db

	updated_weights = np.round(weights, 4).tolist()
	updated_bias = round(float(bias), 4)

	return updated_weights, updated_bias, mse_values