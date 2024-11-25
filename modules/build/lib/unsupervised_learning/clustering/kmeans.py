
import numpy as np
from unsupervised_learning.neighbors import NearestNeighbors
from .clustering import kmeans_plus_plus
from numba import njit, prange
from typing import Literal


def kmeans(X: np.ndarray, k, init, max_iter):
    N = X.shape[0]
    neigh = NearestNeighbors(n_neighbors=1)
    new_centers = np.zeros((k, X.shape[1]))
    if type(init) == np.ndarray:
        centers = init
    elif init == 'kmeans++':
        centers = X[kmeans_plus_plus(X, k)]
    else:
        centers = X[np.random.randint(X.shape[0], size=k)]
    # until stops moving
    for _ in range(max_iter):
        if (new_centers == centers).all():
            break
        else:
            new_centers = np.copy(centers)
            # find_kneighbors(centers, X, 1)
            indices = neigh.fit(centers).kneighbors(X, False)
            centers = recompute_cluster_centers(X, indices, k)
    return centers


@ njit(parallel=True)
def recompute_cluster_centers(X: np.ndarray, indices: np.ndarray, k):
    indices = indices.flatten()
    centers = np.zeros((k, X.shape[1]))
    for cluster_k in prange(k):
        in_cluster_k = np.where(indices == cluster_k)
        nodes = X[in_cluster_k]
        centers[cluster_k] = np.sum(nodes, axis=0) / nodes.shape[0]
    return centers


class KMeans:
    def __init__(self, n_clusters=8, init: Literal['random', 'kmeans++'] = 'kmeans++', max_iter=300):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter

    def fit(self, X: np.ndarray):
        self.centers = kmeans(X, self.n_clusters, self.init, self.max_iter)
        return self

    def predict(self, X: np.ndarray):
        distances, clusters = NearestNeighbors(
            n_neighbors=1).fit(self.centers).kneighbors(X)
        self.loss = np.sum(distances**2)
        return clusters

    def fit_predict(self, X: np.ndarray):
        return self.fit(X).predict(X)
