import numpy as np


def kmeans_plus_plus(X: np.ndarray, k: int):
    X = X ** 2
    centers = np.zeros(k, dtype=int)
    d = np.zeros(X.shape[0], dtype=int)

    centers[0] = np.random.randint(low=X.shape[0])
    # until all clusters are added
    for cluster in range(k):
        for idx, node in enumerate(X):
            closest = np.inf
            # find closest centroid
            for j in range(cluster+1):
                closest = min(
                    abs(np.sum(node - X[centers[j]])), closest)
            d[idx] = closest
        centers[cluster] = np.argmax(np.random.randint(d+1))
    return centers
