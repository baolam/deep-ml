
def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
	# Implement your code here
	tp = tn = fp = fn = 0

	for a, p in zip(actual, predicted):
		if a == 1 and p == 1:
			tp += 1
		elif a == 0 and p == 1:
			fp += 1
		elif a == 0 and p == 0:
			tn += 1
		elif a == 1 and p == 0:
			fn += 1

	confusion_matrix = [
		[tp, fn],
		[fp, tn]
	]

	accuracy = (tp + tn) / len(actual)
	specificity = tn / (tn + fp)
	negativePredictive = tn / (tn + fn)
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	f1 = 2 * precision * recall / (precision + recall)

	return confusion_matrix, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(negativePredictive, 3)
