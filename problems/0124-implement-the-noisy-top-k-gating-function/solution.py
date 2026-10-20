import numpy as np

def noisy_topk_gating(
    X: np.ndarray,
    W_g: np.ndarray,
    W_noise: np.ndarray,
    N: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Args:
        X: Input data, shape (batch_size, features)
        W_g: Gating weight matrix, shape (features, num_experts)
        W_noise: Noise weight matrix, shape (features, num_experts)
        N: Noise samples, shape (batch_size, num_experts)
        k: Number of experts to keep per example
    Returns:
        Gating probabilities, shape (batch_size, num_experts)
    """
    # Your code here
    H_base = np.dot(X, W_g)
    H_noise = np.dot(X, W_noise)

    def softplus(z):
        return np.log(1 + np.exp(z))
    
    H = H_base + N * softplus(H_noise)

    top_k_idx = np.argsort(H, axis=-1)[:, -k:]
    rows = np.arange(H.shape[0])[:, None]

    mask = np.full_like(H, -np.inf, dtype=float)
    mask[rows, top_k_idx] = H[rows, top_k_idx]

    shifted = mask - np.max(mask, axis=-1, keepdims=True)
    exp_H = np.exp(shifted)

    gated_probs = exp_H / np.sum(exp_H, axis=-1, keepdims=True)
    return gated_probs