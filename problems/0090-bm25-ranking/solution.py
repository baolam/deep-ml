import numpy as np
from collections import Counter

def calculate_bm25_scores(corpus, query, k1=1.5, b=0.75):
	N = len(corpus)
	if N == 0:
		return np.array([])

	df = Counter()
	for doc in corpus:
		df.update(set(doc))
	
	doc_lengths = [len(doc) for doc in corpus]
	avg_dl = sum(doc_lengths) / N

	scores = []

	for i, doc in enumerate(corpus):
		score = 0.0
		tf = Counter(doc)
		doc_length = doc_lengths[i]

		for term in query:
			if term not in tf:
				continue
			
			n_t = tf[term]
			idf = np.log((N + 1) / (df[term] + 1))

			numerator = n_t * (k1 + 1)
			denominator = n_t + k1 * (1 - b + b * (doc_length / avg_dl))

			score += idf * (numerator / denominator)
		
		scores.append(score)

	scores = np.array(scores)
	return np.round(scores,3)