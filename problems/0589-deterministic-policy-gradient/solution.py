import numpy as np

def deterministic_policy_gradient(states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray, dones: np.ndarray, theta: np.ndarray, w: np.ndarray, gamma: float, alpha_theta: float, alpha_w: float) -> tuple:
    """
    Perform one update step of the Deterministic Policy Gradient algorithm
    with linear function approximation for both the actor and critic.
    
    Args:
        states: Batch of states, shape (N, state_dim)
        actions: Batch of actions taken, shape (N,)
        rewards: Batch of rewards, shape (N,)
        next_states: Batch of next states, shape (N, state_dim)
        dones: Batch of terminal flags, shape (N,)
        theta: Policy parameters, shape (state_dim,)
        w: Critic parameters, shape (state_dim + 1,)
        gamma: Discount factor
        alpha_theta: Policy learning rate
        alpha_w: Critic learning rate
    
    Returns:
        Tuple of (theta_new, w_new) as lists rounded to 4 decimal places
    """
    N = states.shape[0]

    next_actions = next_states @ theta

    phi_next = np.hstack([next_states, next_actions.reshape(-1, 1)])
    q_next = phi_next @ w

    td_target = rewards + gamma * (1.0 - dones.astype(float)) * q_next
    
    phi_curr = np.hstack([states, actions.reshape(-1, 1)])
    q_curr = phi_curr @ w

    td_error = td_target - q_curr

    grad_w = (phi_curr.T @ td_error) / N
    w_updated = w + alpha_w * grad_w

    dq_da = w[-1]
    grad_theta = dq_da * np.mean(states, axis=0)
    theta_updated = theta + alpha_theta * grad_theta

    return theta_updated, w_updated