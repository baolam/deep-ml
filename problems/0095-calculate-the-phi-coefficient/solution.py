import math

def phi_corr(x: list[int], y: list[int]) -> float:
	"""
	Calculate the Phi coefficient between two binary variables.

	Args:
	x (list[int]): A list of binary values (0 or 1).
	y (list[int]): A list of binary values (0 or 1).

	Returns:
	float: The Phi coefficient rounded to 4 decimal places.
	"""
	# Your code here
	n11 = n00 = n01 = n10 = 0

	for i, j in zip(x, y):
		if i == 1 and j == 1:
			n11 += 1
		elif i == 0 and j == 0:
			n00 += 1
		elif i == 1 and j == 0:
			n10 += 1
		else:
			n01 += 1
	
	numerator = n00 * n11 - n01 * n10
	denominator = math.sqrt((n00 + n01) * (n10 + n11) * (n00 + n10) * (n01 + n11))

	try:
		val = numerator / denominator
	except:
		val = 0.0
	return round(val,4)