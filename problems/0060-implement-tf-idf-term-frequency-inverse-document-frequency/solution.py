import math
import numpy as np

def compute_tf_idf(corpus, query):
	"""
	Compute TF-IDF scores for a query against a corpus of documents.
    
	:param corpus: List of documents, where each document is a list of words
	:param query: List of words in the query
	:return: List of lists containing TF-IDF scores for the query words in each document
	"""
	if len(corpus) == 0 or len(query) == 0:
		return []
	
	N = len(corpus)
	idf = dict()

	for word in query:
		if word in idf:
			continue
		
		df = sum(1 for doc in corpus if word in doc)
		idf[word] = math.log((N + 1) / (df + 1)) + 1
	
	tfidf = []
	for doc in corpus:
		dlen = len(doc)
		d_scores = []

		if dlen == 0:
			d_scores = [0.0] * len(word)
			tfidf.append(d_scores)
			continue
		
		for word in query:
			tf = doc.count(word) / dlen
			score = tf * idf[word]

			d_scores.append(score)

		tfidf.append(d_scores)
	
	return tfidf
		