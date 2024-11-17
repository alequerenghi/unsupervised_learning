from typing import Literal
import numpy as np
from sklearn.neighbors import NearestNeighbors
from unsupervised_learning.clustering.clustering import kmeans_plus_plus
from numba import njit, prange, int64


def kmedoids(X: np.ndarray, k, init, max_iter):
    N = X.shape[0]
    neigh = NearestNeighbors(n_neighbors=1)
    new_centers = np.zeros(k)
    if init == 'kmeans++':
        centers = kmeans_plus_plus(X, k)
    else:
        centers = np.random.randint(X.shape[0], size=k)

    # until stops moving
    for _ in range(max_iter):
        if (new_centers == centers).all():
            break
        else:
            new_centers = np.array(centers, copy=True)
            neigh.fit(X[centers])
            indices = neigh.kneighbors(X, return_distance=False)
            centers = recompute_medoid_centers(X, indices, k)
    return centers


@njit(parallel=True)
def recompute_medoid_centers(X: np.ndarray, indices: np.ndarray, k):
    indices = indices.flatten()
    clusters = np.zeros(k, dtype=int64)
    for cluster in prange(k):
        in_cluster = np.where(indices == cluster)[0]
        nodes = X[in_cluster] ** 2
        dist = np.zeros(len(in_cluster))
        for i in range(dist.shape[0]):
            dist[i] = np.sum(np.sqrt(abs(np.sum(nodes - nodes[i], axis=1))))
        clusters[cluster] = in_cluster[np.argmin(dist)]
    return clusters


class KMedoids:
    def __init__(self, n_clusters=8, init: Literal['random', 'kmeans++'] = 'kmeans++', max_iter=300):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        return self

    def fit(self, X: np.ndarray):
        self.centers = X[kmedoids(X, self.k, self.init, self.max_iter)]
        return self

    def predict(self, X: np.ndarray):
        return NearestNeighbors(n_neighbors=1).fit(self.centers).kneighbors(X, return_distance=False)

    def fit_predict(self, X: np.ndarray):
        return self.fit(X).predict(X)
