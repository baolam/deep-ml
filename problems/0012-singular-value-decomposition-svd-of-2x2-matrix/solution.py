import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 numpy array
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
        - U: 2x2 orthogonal matrix
        - S: length-2 array of singular values
        - Vt: 2x2 orthogonal matrix (transpose of V)
    """
    # Your code here
    B = A.T @ A
    b11, b12, b22 = B[0, 0], B[0, 1], B[1, 1]

    if abs(b12) < 1e-12:
        theta = 0.0
    else:
        theta = 0.5 * np.arctan2(2 * b12, b11 - b22)
    
    c = np.cos(theta)
    s = np.sin(theta)

    V = np.array([
        [c, -s],
        [s, c]
    ])

    A_prime = A @ V

    s0 = np.linalg.norm(A_prime[:, 0])
    s1 = np.linalg.norm(A_prime[:, 1])

    S = np.array([s0, s1])
    if s0 < s1:
        S = np.array([s1, s0])
        V = np.array([
            [-s, c],
            [c, s]
        ])
        A_prime = A @ V
    
    U = np.zeros((2, 2))
    for i in range(2):
        if S[i] > 1e-12:
            U[:, i] = A_prime[:, i] / S[i]
    
    Vt = V.T
    return U, S, Vt