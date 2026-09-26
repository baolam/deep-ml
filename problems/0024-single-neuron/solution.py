import math
import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
	features = np.array(features)
	labels = np.array(labels)
	weights = np.array(weights)

	h = np.dot(features, weights) + bias
	probabilities = 1 / (1 + np.exp(-h))

	mse = ((labels - probabilities) ** 2).mean()
	return probabilities.tolist(), float(mse)