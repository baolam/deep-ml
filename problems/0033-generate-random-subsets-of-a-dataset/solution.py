import numpy as np

def get_random_subsets(X, y, n_subsets, replacements=True):
    X = np.array(X)
    y = np.array(y)

    n_samples = X.shape[0]
    sample_size = n_samples if replacements else n_samples // 2

    subsets = []
    for _ in range(n_subsets):
        idx = np.random.choice(n_samples, size=sample_size, replace=replacements)

        X_sub = X[idx].tolist()
        y_sub = y[idx].tolist()

        subsets.append((X_sub, y_sub))
        
    return subsets