import numpy as np

class LSTM:
	def __init__(self, input_size, hidden_size):
		self.input_size = input_size
		self.hidden_size = hidden_size

		# Initialize weights and biases
		self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

		self.bf = np.zeros((hidden_size, 1))
		self.bi = np.zeros((hidden_size, 1))
		self.bc = np.zeros((hidden_size, 1))
		self.bo = np.zeros((hidden_size, 1))

	def _sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def forward(self, x, initial_hidden_state, initial_cell_state):
		"""
		Processes a sequence of inputs and returns the hidden states, final hidden state, and final cell state.
		"""
		T = x.shape[0]
		h = initial_hidden_state
		c = initial_cell_state

		hidden_states = []
		for t in range(T):
			x_t = x[t].reshape(-1, 1)

			z_t = np.vstack([h, x_t])

			f_t = self._sigmoid(np.dot(self.Wf, z_t) + self.bf)
			i_t = self._sigmoid(np.dot(self.Wi, z_t) + self.bi)
			c_t = np.tanh(np.dot(self.Wc, z_t) + self.bc)
			o_t = self._sigmoid(np.dot(self.Wo, z_t) + self.bo)

			c = f_t * c + i_t * c_t
			h = o_t * np.tanh(c)

			hidden_states.append(h)

		return hidden_states, h, c
		