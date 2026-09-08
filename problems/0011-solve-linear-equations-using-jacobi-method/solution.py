import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
	d = np.diag(A)
	R = A - np.diag(d)

	x = np.zeros_like(b, dtype=float)
	for _ in range(n):
		x = (b - np.dot(R, x)) / d

	return np.round(x, 4).tolist()