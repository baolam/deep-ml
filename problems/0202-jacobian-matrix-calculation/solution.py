import numpy as np

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
	"""
	Compute the Jacobian matrix using numerical differentiation.
	
	Args:
		f: Function that takes a list and returns a list
		x: Point at which to evaluate the Jacobian
		h: Step size for finite differences
	
	Returns:
		Jacobian matrix as list of lists
	"""
	# Your code here
	n = len(x)
	x_arr = np.array(x, dtype=float)
	f0 = np.array(f(x), dtype=float)
	m = len(f0)

	J = np.zeros((m, n), dtype=float)

	for j in range(n):
		xj = x_arr.copy()
		xj[j] += h
		f_new = np.array(
			f(xj.tolist())
		)

		J[:, j] = (f_new - f0) / h
	return J