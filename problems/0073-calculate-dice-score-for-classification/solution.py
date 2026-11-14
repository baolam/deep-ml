
import numpy as np

def dice_score(y_true, y_pred):
    # Write your code here
    n = len(y_true)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if (y_true == 0).sum() == n or (y_pred == 0).sum() == n:
        return 0.0

    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    
    res = 2 * tp / (2 * tp + fp + fn)
    return round(res, 3)
