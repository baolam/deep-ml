import math
import numpy as np

def compute_pmi(joint_counts, total_counts_x, total_counts_y, total_samples):
	# Implement PMI calculation here
	px = total_counts_x / total_samples
	py = total_counts_y / total_samples
	pxy = joint_counts / total_samples
	if pxy == 0:
		return float('-inf')

	pmi = math.log2(pxy / (px * py))
	return np.round(pmi, 3)