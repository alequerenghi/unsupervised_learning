import numpy as np
from unsupervised_learning.neighbors import NearestNeighbors


def kmeans_plus_plus(X: np.ndarray, k: int):
    centers = np.zeros(k, dtype=np.int64)
    neigh = NearestNeighbors(1)
    centers[0] = np.random.randint(low=X.shape[0])
    # until all clusters are added
    for cluster in range(k):
        neigh.fit(X[centers[:cluster+1]])
        distances = neigh.kneighbors(X)[0].flatten()
        distances = distances ** 2 * X.shape[0]+1
        distances.astype(np.int64)
        centers[cluster] = np.argmax(np.random.randint(distances))
    return centers
