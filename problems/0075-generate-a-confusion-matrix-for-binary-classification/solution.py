
from collections import Counter

def confusion_matrix(data):
    # Implement the function here
    mapping = {
        (0, 0) : 0,
        (0, 1) : 0,
        (1, 0) : 0,
        (1, 1) : 0
    }

    for pair in data:
        mapping[(pair[0], pair[1])] += 1

    confuse = [
        [mapping[(1, 1)], mapping[(1, 0)]],
        [mapping[(0, 1)], mapping[(0, 0)]]
    ]

    return confuse