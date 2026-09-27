import numpy as np

def lora_rank_sweep(delta_W_true: list[list[float]], ranks: list[int]) -> list[float]:
	"""
	Compute Frobenius reconstruction error of the best rank-r approximation
	of delta_W_true for each r in ranks.

	Args:
		delta_W_true: target weight-update matrix, shape (m, n)
		ranks: list of candidate ranks to evaluate

	Returns:
		List of reconstruction errors, one per rank, in the same order as `ranks`.
	"""
	W = np.array(delta_W_true, dtype=np.float64)
	s = np.linalg.svd(W, compute_uv=False)

	errors = []
	for r in ranks:
		val = 0.0
		if r <= 0:
			val = np.sqrt(np.sum(s ** 2))
		elif r < len(s):
			val = np.sqrt(np.sum(s[r:] ** 2))
		errors.append(float(val))

	return errors