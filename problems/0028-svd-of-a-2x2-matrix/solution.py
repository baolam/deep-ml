import numpy as np

def svd_2x2(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix.
    
    Args:
        A: 2x2 numpy array
    
    Returns:
        U: 2x2 orthogonal matrix (left singular vectors)
        s: 1D array of singular values
        V: 2x2 matrix (right singular vectors)
    """
    # Your code here
    A = np.array(A, dtype=float)
    ATA = A.T @ A

    val, vec = np.linalg.eigh(ATA)

    idx = np.argsort(val)[::-1]
    val = np.maximum(val[idx], 0.0)

    V_cols = vec[:, idx]
    s = np.sqrt(val)
    V = V_cols.T

    U = np.zeros((2, 2), dtype=float)
    tol = 1e-12

    for i in range(2):
        if s[i] > tol:
            U[:, i] = (A @ V[i, :]) / s[i]
        else:
            if i == 1:
                U[:, 1] = np.array([-U[1, 0], U[0, 0]])
            else:
                U[:, 0] = np.array([1.0, 0.0])
    
    return U, s, V