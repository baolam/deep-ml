import numpy as np

def rejection_sampling_best_of_k(candidates, scores):
    """
    Select the highest-scoring candidate per prompt.

    Args:
        candidates: list of N lists, each containing K candidate outputs.
        scores: list of N lists, each containing K reward scores.

    Returns:
        List of N selected candidates.
    """
    ans = []

    for candidate, score in zip(candidates, scores):
        idx = np.argmax(np.array(score))
        ans.append(candidate[idx])

    return ans