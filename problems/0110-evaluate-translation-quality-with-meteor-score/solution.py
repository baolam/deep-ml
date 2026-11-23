import numpy as np
from collections import Counter

def meteor_score(reference, candidate, alpha=0.9, beta=3, gamma=0.5):
    """
    Calculate METEOR score for machine translation evaluation.
    
    Args:
        reference: Reference translation string
        candidate: Candidate translation string
        alpha: Weight for precision vs recall in F-mean (default 0.9)
        beta: Exponent for fragmentation penalty (default 3)
        gamma: Maximum penalty coefficient (default 0.5)
    
    Returns:
        METEOR score between 0 and 1
    """
    # Your code here
    ref = reference.lower().split()
    can = candidate.lower().split()

    if not ref or not can:
        return 0.0
    
    used_ref = [False] * len(ref)
    alignments = []

    for cand_idx, cand_word in enumerate(can):
        for ref_idx, ref_word in enumerate(ref):
            if not used_ref[ref_idx] and cand_word == ref_word:
                used_ref[ref_idx] = True
                alignments.append((cand_idx, ref_idx))
                break

    m = len(alignments)
    precision = m / len(can)
    recall = m / len(ref)

    f_mean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)

    chunks = 1
    for i in range(1, m):
        prev_can, prev_ref = alignments[i - 1]
        curr_can, curr_ref = alignments[i]
        if prev_can + 1 != curr_can or prev_ref + 1 != curr_ref:
            chunks += 1
    
    penalty = gamma * ((chunks / m) ** beta)
    meteor = f_mean * (1 - penalty)
    return round(float(meteor), 3)