import numpy as np

def mish(x: float) -> float:
    """
    Compute the Mish activation function.

    Args:
        x (float): Input value

    Returns:
        float: Mish activation value rounded to 4 decimal places
    """
    val1 = np.log(1 + np.exp(x))
    val2 = x * np.tanh(val1)
    return np.round(val2, 4)