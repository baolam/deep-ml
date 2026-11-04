import numpy as np
def q_learning(num_states, num_actions, P, R, terminal_states, alpha, gamma, epsilon, num_episodes):
    # Your code here
    Q = np.zeros((num_states, num_actions), dtype=float)

    terminal_set = set(terminal_states)
    non_terminal_states = [s for s in range(num_states) if s not in terminal_set]

    if not non_terminal_states:
        return Q
    
    for _ in range(num_episodes):
        state = np.random.choice(non_terminal_states)

        while state not in terminal_states:
            if np.random.rand() < epsilon:
                action = np.random.randint(num_actions)
            else:
                action = int(np.argmax(Q[state]))

            next_state = np.random.choice(num_states, p=P[state, action])

            r = R[state, action]
            if next_state in terminal_states:
                target = r
            else:
                target = r + gamma * np.max(Q[next_state])
            
            Q[state, action] += alpha * (target - Q[state, action])
            state = next_state
            
    return Q