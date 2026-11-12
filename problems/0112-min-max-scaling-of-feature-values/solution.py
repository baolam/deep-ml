import numpy as np

def min_max(x: list[float]) -> list[float]:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A list of numerical values
    
    Returns:
        A new list with values normalized to [0, 1]
    """
    # Your code here
    x = np.array(x, dtype=float)
    maxx = np.max(x, keepdims=False)
    minx = np.min(x, keepdims=False)
    if maxx == minx:
        return np.zeros_like(x.shape, dtype=float)
    return (x - minx) / (maxx - minx)