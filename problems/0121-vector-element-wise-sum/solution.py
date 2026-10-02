def vector_sum(a: list[int|float], b: list[int|float]) -> list[int|float]:
	# Return the element-wise sum of vectors 'a' and 'b'.
	# If vectors have different lengths, return -1.
	if len(a) != len(b):
		return -1
	
	c = [0] * len(a)
	for i, (x, y) in enumerate(zip(a, b)):
		c[i] = x + y

	return c