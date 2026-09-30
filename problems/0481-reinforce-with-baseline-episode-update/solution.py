import numpy as np

def reinforce_baseline_update(episode: list, theta: np.ndarray, w: np.ndarray, gamma: float, alpha_theta: float, alpha_w: float) -> dict:
	"""
	Perform REINFORCE with baseline update for a single episode.
	
	Args:
		episode: list of (state, action, reward) tuples
		theta: policy parameters, shape (n_features, n_actions)
		w: baseline parameters, shape (n_features,)
		gamma: discount factor
		alpha_theta: learning rate for policy
		alpha_w: learning rate for baseline
	
	Returns:
		dict with 'theta', 'w', 'returns', 'advantages'
	"""
	T = len(episode)
	n_features, n_actions = theta.shape

	theta_updated = theta.copy()
	w_updated = w.copy()

	returns = np.zeros(T)
	G = 0.0
	for t in reversed(range(T)):
		_, _, reward = episode[t]
		G = reward + gamma * G
		returns[t] = G
	
	advantges = np.zeros(T)
	for t in range(T):
		state, action, _ = episode[t]
		state = np.array(state, dtype=np.float64)

		G_t = returns[t]
		v_s = np.dot(state, w_updated)
		delta = G_t - v_s
		advantges[t] = delta

		logits = np.dot(state, theta_updated)
		exp_logits = np.exp(logits - np.max(logits))
		probs = exp_logits / np.sum(exp_logits)

		d_softmax = -probs
		d_softmax[action] += 1.0
		grad_log_pi = np.outer(state, d_softmax)

		w_updated += alpha_w * delta * state
		theta_updated += alpha_theta * (gamma ** t) * delta * grad_log_pi
	
	return {
		"theta" : theta_updated,
		"w" : w_updated,
		"returns" : returns,
		"advantages" : advantges
	}