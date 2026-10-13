import numpy as np

def batch_iterator(X, y=None, batch_size=64):
	# Your code here
	X = np.array(X)
	n_samples = X.shape[0]

	for i in range(0, n_samples, batch_size):
		begin = i
		end = min(i + batch_size, n_samples)

		if y is None:
			yield X[begin:end].tolist()
		else:
			yield [X[begin:end].tolist(), y[begin:end].tolist()]