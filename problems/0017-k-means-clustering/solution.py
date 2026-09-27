import numpy as np

def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	points = np.array(points)
	centroids = np.array(initial_centroids)

	for _ in range(max_iterations):
		distances = np.linalg.norm(points[:, np.newaxis, :] - centroids, axis=2)
		labels = np.argmax(distances, axis=1)

		new_centroids = np.zeros_like(centroids, dtype=float)
		for i in range(k):
			cluster = points[labels == i]
			if len(cluster) > 0:
				new_centroids[i] = np.mean(cluster, axis=0)
			else:
				new_centroids[i] = cluster[i]
		
		if np.all(new_centroids == centroids):
			break

		centroids = new_centroids
	
	return [
		tuple(round(c, 4) for c in val)
		for val in centroids.tolist()
	]