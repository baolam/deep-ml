from collections import Counter

def disorder(apples: list) -> float:
    """
    Compute the disorder in a basket of apples.
    """
    # Your code here
    n = len(apples)
    stats = Counter(apples)

    res = 0
    for xi in stats.values():
        res += (xi / n) ** 2

    return 1 - res