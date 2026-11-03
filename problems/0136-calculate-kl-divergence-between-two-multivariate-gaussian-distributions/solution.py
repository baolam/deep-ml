import numpy as np

def multivariate_kl_divergence(mu_p: np.ndarray, Cov_p: np.ndarray, mu_q: np.ndarray, Cov_q: np.ndarray) -> float:
    """
    Computes the KL divergence between two multivariate Gaussian distributions.
    
    Parameters:
    mu_p: mean vector of the first distribution
    Cov_p: covariance matrix of the first distribution
    mu_q: mean vector of the second distribution
    Cov_q: covariance matrix of the second distribution

    Returns:
    KL divergence as a float
    """
    # Your code here
    k = mu_p.shape[0]

    trace_term = np.trace(np.linalg.solve(Cov_q, Cov_p))
    diff = mu_q - mu_p
    quad = diff.T @ np.linalg.solve(Cov_q, diff)

    _, log_det_q = np.linalg.slogdet(Cov_q)
    _, log_det_p = np.linalg.slogdet(Cov_p)
    log_det_term = log_det_q - log_det_p

    kl = 0.5 * (trace_term + quad - k + log_det_term)
    return kl