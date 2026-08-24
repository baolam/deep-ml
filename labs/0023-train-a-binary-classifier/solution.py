import numpy as np
# You can import any sklearn module you need

def train(X_train, y_train, X_val, y_val):
    """
    Train a binary classifier.
    
    Args:
        X_train: numpy array of shape (n_samples, 30) -- standardized features
        y_train: numpy array of shape (n_samples,) -- binary labels (0 or 1)
        X_val:   numpy array of shape (n_val, 30) -- standardized
        y_val:   numpy array of shape (n_val,) -- validation labels
    
    Returns:
        predict: callable that takes X (n, 30) and returns y_pred (n,) of 0s and 1s
    """
    # TODO: train a model and return a predict function
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    N, n_features = X_train.shape

    def _get_W(X, y_hat, y, W):
        return  W - 0.001 * np.dot(X.T, y_hat - y) / N
    
    def _get_b(y_hat, y, b):
        return b - 0.001 * (y_hat - y).mean()
    
    w = np.zeros(n_features, dtype=np.float64)
    b = 0.0

    for _ in range(1000):
        z = np.dot(X_train, w) + b
        y_hat = _sigmoid(z)

        w = _get_W(X_train, y_hat, y_train, w)
        b = _get_b(y_hat, y_train, b)
    
    def predict(X):
        z = np.dot(X, w) + b
        y_hat = _sigmoid(z)
        return (y_hat >= 0.5).astype(int)
    
    return predict
