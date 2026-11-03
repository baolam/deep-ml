import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    # Your code here
    m, n = X.shape
    
    b_size = batch_size
    if method == "batch":
        b_size = m
    elif method == "stochastic":
        b_size = 1
    
    w = np.copy(weights).astype(float)
    for _ in range(n_epochs):
        for i in range(0, m, b_size):
            X_batch = X[i:i + b_size]
            y_batch = y[i:i + b_size]

            current_b_size = len(y_batch)

            y_pred = X_batch.dot(w)
            err = y_pred - y_batch

            grad = (2.0 / current_b_size) * X_batch.T.dot(err)

            w = w - learning_rate * grad

    return w            