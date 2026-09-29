import math
import numpy as np

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    scores = np.array(scores)
    cal = scores - np.max(scores)

    num = np.exp(cal)
    dem = np.sum(num)

    return (num / dem).tolist()