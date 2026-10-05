import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    # Your code here
    index = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(index)
    size = n_samples // k
    remain = n_samples % k

    folds = []
    start, end = 0, 0
    for i in range(k):
        fold_size = size + 1 if i < remain else size 
        end = start + fold_size
        folds.append(index[start:end])
        start = end
    
    outputs = []
    for i in range(k):
        test = folds[i].tolist()
        train = []
        for j in range(k):
            if j == i: continue
            train += folds[j].tolist()

        outputs.append((train, test))

    return outputs