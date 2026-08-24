def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	import numpy as np
	matrix = np.array(matrix)

	axis = 0 if mode == 'column' else 1
	means = np.mean(matrix, axis=axis)
	means = means.tolist()
	
	return means