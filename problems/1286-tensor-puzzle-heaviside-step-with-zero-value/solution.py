import numpy as np

def heaviside(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise heaviside with zero-value b."""
    out = 1 * (a > 0) + b * (a == 0)
    return out