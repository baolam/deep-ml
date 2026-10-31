def calculate_f1_score(y_true, y_pred):
	"""
	Calculate the F1 score based on true and predicted labels.

	Args:
		y_true (list): True labels (ground truth).
		y_pred (list): Predicted labels.

	Returns:
		float: The F1 score rounded to three decimal places.
	"""
	# Your code here
	tp = fp = tn = fn = 0

	for true, pred in zip(y_true, y_pred):
		if true == 1 and pred == 1: tp += 1
		if true == 0 and pred == 0: tn += 1
		if pred == 0 and true == 1: fp += 1
		if pred == 1 and true == 0: fn += 1

	try:
		precision = tp / (tp + fp)
		recall = tp / (tp + fn)
		f1 = 2 * precision * recall / (precision + recall)
	except:
		f1 = 0.0
	return round(f1,3)