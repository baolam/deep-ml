import numpy as np

def cosine_similarity(v1, v2):
	"""
	Calculate the cosine_similarity of two vectors.
	Args:
		vec1 (numpy.ndarray): 1D array representing the first vector.
		vec2 (numpy.ndarray): 1D array representing the second vector.
	Returns:
		The cosine_similarity of the two vectors.
	"""
	# Implement your code here
	v1 = np.array(v1, dtype=float)
	v2 = np.array(v2, dtype=float)

	a = (v1 * v2).sum() / (np.linalg.norm(v1) * np.linalg.norm(v2))
	return a