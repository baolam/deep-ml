import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    # Your code here
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    normX = (data - mean) / std
    cov = np.cov(normX, rowvar=False)
    values, vectors = np.linalg.eigh(cov)

    sort_indices = np.argsort(values)[::-1]
    components = vectors[:, sort_indices]
    components = components[:, :k]

    for j in range(k):
        col = components[:, j]
        non_zindex = np.where(np.abs(col) > 1e-10)[0]
        if len(non_zindex) <= 0:
            continue
        first = col[non_zindex[0]]
        if first < 0:
            components[:, j] *= -1.0
 
    return np.round(components, 4)
    