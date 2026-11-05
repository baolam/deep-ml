import numpy as np

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
	"""
	Compute entropy of P and cross-entropy between P and Q.
	
	Args:
		P: True probability distribution
		Q: Predicted probability distribution
	
	Returns:
		Tuple of (entropy H(P), cross-entropy H(P,Q))
	"""
	# Your code here
	hp, hpq = 0, 0
	is_inf = False

	for p, q in zip(P, Q):
		if p > 0:
			hp += p * np.log(p)
			if q == 0:
				is_inf = True
			elif q > 0:
				hpq += p * np.log(q)

	return -hp, float('inf') if is_inf else -hpq