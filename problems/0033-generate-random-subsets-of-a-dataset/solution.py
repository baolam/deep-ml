import numpy as np

def get_random_subsets(X, y, n_subsets, replacements=True):
    # Your code here
    X = np.array(X)
    y = np.array(y)

    n_samples = X.shape[0]
    sample_size = n_samples if replacements else n_samples // 2

    subsets = []

    for _ in range(n_subsets):
        idx = np.random.choice(n_samples, size=sample_size, replace=replacements)

        X_subset = X[idx].tolist()
        y_subset = y[idx].tolist()

        subsets.append((X_subset, y_subset))
        
    return subsets