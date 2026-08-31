import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
	A = np.array(A)
	T = np.array(T)
	S = np.array(S)

	is_not_ok = np.isclose(np.linalg.det(T), 0) or np.isclose(np.linalg.det(S), 0)
	if is_not_ok:
		return -1

	T_inv = np.linalg.inv(T)
	transformed = T_inv @ A @ S

	return transformed.tolist()

