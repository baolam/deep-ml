import numpy as np

def accuracy_score(y_true, y_pred):
	# Your code here
	n = len(y_true)
	c = (np.array(y_true) == np.array(y_pred)).sum()
	return c / n