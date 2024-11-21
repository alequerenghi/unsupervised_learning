import numpy as np
from sklearn.neighbors import NearestNeighbors
from numba import njit, int64


@njit
def kmeans_plus_plus(X: np.ndarray, k: int):
    centers = np.zeros(k, dtype=int64)
    d = np.zeros(X.shape[0], dtype=int64)

    centers[0] = np.random.randint(low=X.shape[0])
    # until all clusters are added
    for cluster in range(k):
        distances = NearestNeighbors(n_neighbors=1).fit(
            X[centers[:cluster+1]]).kneighbors(X)[0].flatten()
        distances.astype(int64)
        for idx, val in enumerate(distances):
            d[idx] = np.random.randint(val**2+1)
        centers[cluster] = np.argmax(d)
    return centers


def compute_loss(clusters: np.ndarray, centers: np.ndarray, n_clusters):
    mean = clusters.mean()
    std = clusters.std()
    clusters = (clusters-mean)/std
    centers = (centers-mean)/std
    clusters = clusters ** 2
    loss = 0
    for cluster in range(n_clusters):
        in_cluster = np.where(clusters == cluster)[0]
        loss += np.sum(
            np.sqrt(np.abs(np.sum(clusters[in_cluster] - centers[cluster] ** 2, axis=1))))
    return loss
