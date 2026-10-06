import numpy as np
def translate_object(points, tx, ty):
	points = np.array(points, dtype=float)
	trans = np.array([[tx, ty]], dtype=float)

	translated_points = points + trans

	return translated_points.tolist()
