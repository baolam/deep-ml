
import numpy as np
import sympy as sp

def matrix_image(A):
	# Write your code here
	M = sp.Matrix(A)

	rref, pivot = M.rref()

	A = np.array(A, dtype=float)
	col_space = A[:, pivot]

	return col_space