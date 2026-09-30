import numpy as np

def gpt_feedforward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """GPT-style position-wise feedforward block.

    Args:
        x:  array of shape (batch_size, num_tokens, emb_dim)
        W1: array of shape (emb_dim, 4*emb_dim)
        b1: array of shape (4*emb_dim,)
        W2: array of shape (4*emb_dim, emb_dim)
        b2: array of shape (emb_dim,)

    Returns:
        Array of shape (batch_size, num_tokens, emb_dim).
    """
    def _linear(X, W, b):
        return X @ W + b
    
    def _gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * (x ** 3))))
    
    h1 = _linear(x, W1, b1)
    a1 = _gelu(h1)
    h2 = _linear(a1, W2, b2)
    return h2
