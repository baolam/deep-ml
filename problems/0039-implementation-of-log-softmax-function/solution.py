import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	# Your code here
	scores = np.array(scores)
	max_score = np.max(scores, keepdims=True)
	prob = np.exp(scores - max_score)

	return scores - max_score - np.log(prob.sum())