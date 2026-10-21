import numpy as np
def simulate_markov_chain(transition_matrix, initial_state, num_steps):
    # Your code here
    num_states = transition_matrix.shape[0]

    states = [initial_state]

    for _ in range(num_steps):
        probs = transition_matrix[states[-1]]

        state = np.random.choice(num_states, p=probs)

        states.append(state)

    return states