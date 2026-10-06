import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
	input_height, input_width = input_matrix.shape
	kernel_height, kernel_width = kernel.shape

	padded_input = np.pad(
		input_matrix,
		pad_width=padding,
		mode='constant',
		constant_values=0
	)	

	output_height = (input_height + 2 * padding - kernel_height) // stride + 1
	output_width = (input_width + 2 * padding - kernel_width) // stride + 1

	output_matrix = np.zeros((output_height, output_width))

	for i in range(output_height):
		for j in range(output_width):
			h_start = i * stride
			h_end = h_start + kernel_height
			w_start = j * stride
			w_end = w_start + kernel_width

			region = padded_input[h_start:h_end, w_start:w_end]
			output_matrix[i, j] = np.sum(region * kernel)
			
	return output_matrix
