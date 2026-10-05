import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
	# Your code here
	# input_sequence = np.array(input_sequence)
	initial_hidden_state = np.array(initial_hidden_state)
	Wx = np.array(Wx)
	Wh = np.array(Wh)
	b = np.array(b)

	h_t = initial_hidden_state
	for x_t in input_sequence:
		x_t = np.array(x_t)
		h_t = np.tanh(Wx @ x_t + Wh @ h_t + b)
	return h_t