import numpy as np
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    # ✏️  Your code here
    n_samples, n_features = X.shape

    combinations = []
    for d in range(degree + 1):
        combinations.extend(
            combinations_with_replacement(range(n_features), d)
        )
    
    n_output = len(combinations)
    X_poly = np.empty((n_samples, n_output))

    for i, combo in enumerate(combinations):
        X_poly[:, i] = np.prod(X[:, combo], axis=-1)

    X_poly = np.sort(X_poly, axis=-1)
    return X_poly