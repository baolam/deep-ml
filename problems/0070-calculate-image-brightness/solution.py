import numpy as np

def calculate_brightness(img):
	# Write your code here
	if len(img) == 0:
		return -1

	try:
		img = np.array(img)
		m, n = img.shape
		if m != n:
			return -1

		return img.mean()
	except:
		return -1
