import numpy as np

def grammar_constrained_step(logits, allowed, transitions, current_state, sampled_token):
    # logits: list of floats
    # allowed: list/set of allowed token indices
    # transitions: dict mapping (state, token) -> next_state
    # current_state: current FSM state
    # sampled_token: index of the sampled token
    logits = np.array(logits)
    probs = np.zeros_like(logits)

    allowed_set = set(allowed)
    if allowed_set:
        allowed_indicies = list(allowed_set)
        allowed_logits = logits[allowed_indicies]

        exp_logits = np.exp(allowed_logits - np.max(allowed_logits))
        softmax_probs = exp_logits / np.sum(exp_logits)
    
        probs[allowed_indicies] = softmax_probs
    
    new_state = transitions.get((current_state, sampled_token), current_state)
    return probs.tolist(), new_state