import numpy as np

def pairwise_cosine_similarity(X):
    # Your code here
    X = np.array(X)
    dot = X @ X.T
    norms = np.linalg.norm(X, axis=1)
    norms = np.where(norms == 0, 1.0, norms)
    denom = np.outer(norms, norms)
    sim = dot / denom
    return np.round(sim, 4).tolist()