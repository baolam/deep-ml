import torch

def mse(pred, target):
    """
    Compute mean squared error between pred and target.

    Args:
        pred (torch.Tensor): Predicted values.
        target (torch.Tensor): Ground-truth values (same shape as pred).

    Returns:
        float: Mean of squared differences.
    """
    return ((target - pred) ** 2).mean().item()
