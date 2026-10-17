import numpy as np

def f_score(y_true, y_pred, beta):
	"""
	Calculate F-Score for a binary classification task.

	:param y_true: Numpy array of true labels
	:param y_pred: Numpy array of predicted labels
	:param beta: The weight of precision in the harmonic mean
	:return: F-Score rounded to three decimal places
	"""
	y_true = np.array(y_true, dtype=float)
	y_pred =  np.array(y_pred, dtype=float)

	tp = ((y_true == 1) & (y_pred == 1)).sum()
	fp = ((y_true == 0) & (y_pred == 1)).sum()
	fn = ((y_true == 1) & (y_pred == 0)).sum()

	prec = tp / (tp + fp)
	rec = tp / (tp + fn)

	f_score = (1 + beta ** 2) * (prec * rec) / ((beta ** 2 * prec) + rec)
	return round(f_score, 3)	