import numpy as np
from collections import Counter

def calculate_bm25_scores(corpus, query, k1=1.5, b=0.75):
	# Your code here
	N = len(corpus)
	if N == 0:
		return np.array([])

	doc_lengths = [len(doc) for doc in corpus]
	avg_dl = sum(doc_lengths) / N

	df = Counter()
	for doc in corpus:
		df.update(set(doc))

	scores = []

	for i, doc in enumerate(corpus):
		doc_len = doc_lengths[i]
		tf = Counter(doc)

		score = 0.0

		for term in query:
			if term not in tf:
				continue
			
			freq = tf[term]
			n_t = df[term]

			# if N - n_t <= 0:
			# 	idf = 0.0
			# else:
			# 	idf = max(0.0, np.log((N - n_t) / n_t))
			idf = np.log((N + 1) / (n_t + 1))

			nume = freq * (k1 + 1)
			deno = freq + k1 * (1 - b + b * (doc_len / avg_dl))

			score += idf * (nume / deno)
		
		scores.append(score)
	
	scores = np.array(scores)
	return np.round(scores,3)