import numpy as np

def masked_ce_loss(logits: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute mean cross-entropy loss over masked (response) positions only.

    Args:
        logits: (seq_len, vocab_size) array of unnormalized scores.
        targets: (seq_len,) array of integer target token ids.
        mask: (seq_len,) boolean array; True = include in loss.

    Returns:
        Mean cross-entropy over positions where mask is True (float).
    """
    max_logits = np.max(logits, axis=-1, keepdims=True)
    cal_logits = logits - max_logits
    log_sum_exp = max_logits + np.log(np.sum(np.exp(cal_logits), axis=-1, keepdims=True))
    log_sum_exp = log_sum_exp.squeeze(-1)

    seq_len = logits.shape[0]
    target_logits = logits[np.arange(0, seq_len, 1), targets]

    ce_per_token = log_sum_exp - target_logits
    masked_loss = ce_per_token[mask]

    if len(masked_loss) == 0:
        return  0.0

    return float(np.mean(masked_loss))
