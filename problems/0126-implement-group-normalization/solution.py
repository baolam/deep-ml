import numpy as np

def group_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, num_groups: int, epsilon: float = 1e-5) -> np.ndarray:
    # Your code here
    B, C, H, W = X.shape
    G = num_groups

    X_grouped = X.reshape(B, G, C // G, H, W)

    mean = X_grouped.mean(axis=(2, 3, 4), keepdims=True)
    var = X_grouped.var(axis=(2, 3, 4), keepdims=True)

    X_normed = (X_grouped - mean) / np.sqrt(var + epsilon)
    X_normed = X_normed.reshape(B, C, H, W)

    gamma = gamma.reshape(1, C, 1, 1)
    beta = beta.reshape(1, C, 1, 1)

    return gamma * X_normed + beta
