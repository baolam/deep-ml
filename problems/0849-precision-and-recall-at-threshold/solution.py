import numpy as np

def precision_recall_at_threshold(y_true, y_scores, threshold):
    """
    Compute precision and recall at a given decision threshold.

    Args:
        y_true: list/array of true binary labels (0 or 1)
        y_scores: list/array of predicted scores in [0, 1]
        threshold: float, classification threshold (predict positive if score >= threshold)

    Returns:
        [precision, recall] as a list of two floats rounded to 4 decimals.
    """
    # Your code here
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    pred = (y_scores >= threshold).astype(int)

    tp = ((pred == 1) & (y_true == 1)).sum()
    actual_p = (y_true == 1).sum()
    pred_p = (pred == 1).sum()

    precision = float(tp / pred_p) if pred_p > 0 else 0.0
    recall = float(tp / actual_p) if actual_p > 0 else 0.0

    return [round(precision, 4), round(recall, 4)]
