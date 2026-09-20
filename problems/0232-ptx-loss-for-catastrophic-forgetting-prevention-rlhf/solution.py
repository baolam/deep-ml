import numpy as np

def compute_ptx_loss(
    rl_loss: float,
    pretrain_logits: np.ndarray,
    pretrain_labels: np.ndarray,
    beta_ptx: float = 0.1
) -> tuple[float, float, float]:
	"""
	Compute PTX (Pre-training) Loss to prevent catastrophic forgetting in RLHF.
	
	PTX Loss = RL Loss + beta_ptx * Cross-Entropy Loss
	
	Prevents model from forgetting general capabilities while
	fine-tuning with reinforcement learning from human feedback.
	
	Args:
		rl_loss: Reinforcement learning loss (e.g., PPO objective)
		pretrain_logits: Model logits on pre-training batch
		  Shape: (batch_size, vocab_size)
		pretrain_labels: True token indices
		  Shape: (batch_size,)
		beta_ptx: Weight coefficient (typically 0.05-0.2)
	
	Returns:
		Tuple of (total_loss, ce_loss, weighted_ce_loss):
		- total_loss: L_RL + beta_ptx * L_CE
		- ce_loss: Cross-entropy on pre-training data
		- weighted_ce_loss: beta_ptx * L_CE
	"""
	# Your code here
	logits = np.max(pretrain_logits, axis=-1, keepdims=True)
	shifted = pretrain_logits - logits

	log_softmax = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
	log_probs = shifted - log_softmax

	batch_size = len(pretrain_logits)
	batch_index = np.arange(0, batch_size, 1)

	true = log_probs[batch_index, pretrain_labels]

	ce_loss = float(-np.mean(true))
	weighted_ce = beta_ptx * ce_loss
	total_loss = rl_loss + weighted_ce

	return total_loss, ce_loss, weighted_ce