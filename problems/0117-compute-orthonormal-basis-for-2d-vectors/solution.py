import numpy as np

def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[np.ndarray]:
    # Your code here
    basis = []

    for v in vectors:
        w = np.array(v, dtype=float)

        for u in basis:
            proj = np.dot(w, u) * u
            w -= proj

        norm = np.linalg.norm(w)
        if norm > tol:
            w = w / norm
            basis.append(w)
            
    return basis