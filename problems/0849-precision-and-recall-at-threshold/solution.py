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
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    y_pred = (y_scores >= threshold).astype(int)

    tp = ((y_pred == 1) & (y_true) == 1).sum()
    a_p = (y_true == 1).sum()
    p_p = (y_pred == 1).sum()

    precision = float(tp / p_p) if p_p > 0 else 0.0
    recall = float(tp / a_p) if a_p > 0 else 0.0

    return [round(precision, 4), round(recall, 4)]