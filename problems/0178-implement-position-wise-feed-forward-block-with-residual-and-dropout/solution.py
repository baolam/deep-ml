import numpy as np

def ffn(x: list[float], W1: list[list[float]], b1: list[float], W2: list[list[float]], b2: list[float], dropout_p: float=0.1, seed: int=42) -> list[float]:
	"""
	Implement a position-wise feed-forward block with residual and dropout.

	Args:
		x: input vector
		W1, b1: first linear layer parameters
		W2, b2: second linear layer parameters
		dropout_p: dropout probability
		seed: random seed for reproducibility

	Returns:
		Output vector after FFN block (rounded to 4 decimals)
	"""
	# Your code here
	x = np.array(x)
	W1 = np.array(W1)
	b1 = np.array(b1)
	W2 = np.array(W2)
	b2 = np.array(b2)

	h1 = np.dot(W1, x) + b1
	a1 = np.maximum(0, h1)

	h2 = np.dot(W2, a1) + b2
	if dropout_p > 0:
		np.random.seed(seed)
		mask = (np.random.rand(*h2.shape) >= dropout_p) / (1 - dropout_p)
		h2 = h2 * mask
	
	h2 = h2 + x
	return h2.tolist()
