import numpy as np

def k_nearest_neighbors(points, query_point, k):

    dist = []

    for i, point in enumerate(points):
        distance = np.linalg.norm(
            np.array(point) - np.array(query_point)
        )
        dist.append((distance, i))

    dist.sort()

    return [points[i] for d, i in dist[:k]]