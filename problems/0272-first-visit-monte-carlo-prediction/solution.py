import numpy as np

def first_visit_mc_prediction(
    episodes: list[list[tuple[int, float]]],
    n_states: int,
    gamma: float
) -> np.ndarray:
    """
    Estimate state values using First-Visit Monte Carlo prediction.
    
    Args:
        episodes: List of episodes. Each episode is a list of (state, reward) tuples.
                 The reward at index i is the reward received AFTER leaving state i.
        n_states: Number of states (states are integers 0 to n_states-1)
        gamma: Discount factor
        
    Returns:
        V: Estimated state values as numpy array of shape (n_states,)
    """
    return_sums = np.zeros(n_states, dtype=float)
    return_counts = np.zeros(n_states, dtype=float)

    for episode in episodes:
        first = {}

        for t, (state, reward) in enumerate(episode):
            if state not in first:
                first[state] = t
        
        G = 0.0
        for t in range(len(episode) - 1, -1, -1):
            state, reward = episode[t]
            G = reward + gamma * G

            if first[state] == t:
                return_sums[state] += G
                return_counts[state] += 1.0
    
    V = np.zeros(n_states, dtype=float)
    visited_mask = return_counts > 0

    V[visited_mask] = return_sums[visited_mask] / return_counts[visited_mask]
    return V 