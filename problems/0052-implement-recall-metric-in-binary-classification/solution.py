import numpy as np

def recall(y_true, y_pred):
    """
    Calculate the recall metric for binary classification.
    
    Args:
        y_true: Array of true binary labels (0 or 1)
        y_pred: Array of predicted binary labels (0 or 1)
    
    Returns:
        Recall value as a float
    """
    # Your code here
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = ((y_true == 1) & (y_pred == 1)).sum()
    # tn = (y_true == 0 & y_pred == 0).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    # true, pred
    # tp (1, 1)   fp (0, 1)
    # fn (1, 0)   tn (0, 0)


    # Recall, positive / actual_positive
    ans = tp / (tp + fn)
    return ans
