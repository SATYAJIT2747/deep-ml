import numpy as np

def k_means_clustering(
    points: list[tuple[float, ...]],
    k: int,
    initial_centroids: list[tuple[float, ...]],
    max_iterations: int
) -> list[tuple[float, ...]]:

    centroids = initial_centroids

    for _ in range(max_iterations):

        # Create k empty clusters
        clusters = [[] for _ in range(k)]

        # Assign points to nearest centroid
        for point in points:

            distances = []

            for i, center in enumerate(centroids):
                dist = np.linalg.norm(
                    np.array(point) - np.array(center)
                )
                distances.append((dist, i))

            # Find nearest centroid
            _, cluster_idx = min(distances)

            # Add point to that cluster
            clusters[cluster_idx].append(point)

        # Calculate new centroids
        new_centroids = []

        for i, cluster in enumerate(clusters):

            if cluster:
                centroid = tuple(np.mean(cluster, axis=0))
            else:
                # Keep old centroid if cluster is empty
                centroid = centroids[i]

            new_centroids.append(centroid)

        # Update centroids
        centroids = new_centroids

    return centroids