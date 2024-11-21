

import numpy as np
from sklearn.neighbors import NearestNeighbors
from clustering import kmeans_plus_plus, compute_loss
from numba import njit, prange
from typing import Literal


def fuzzy_cmeans(X: np.ndarray, k, fuzz, init, max_iter):
    N = X.shape[0]
    new_centers = np.zeros((k, X.shape[1]))
    U = np.zeros((N, k))
    if init == 'kmeans++':
        centers = X[kmeans_plus_plus(X, k)]
    elif type(init) == np.ndarray:
        centers = init
    else:
        centers = X[np.random.randint(X.shape[0], size=k)]
    # until stops moving
    for _ in range(max_iter):
        if (new_centers == centers).all():
            break
        else:
            new_centers = np.copy(centers)
            centers = recompute_cluster_centers(X, U, fuzz)
            U = recompute_fuzz(X, centers, fuzz)
    return centers


def recompute_fuzz(X: np.ndarray, centers: np.ndarray, fuzz):
    X **= 2
    centers **= 2
    N = X.shape[0]
    k = centers.shape[0]
    center_distances = np.zeros((N, k))
    for cluster_k, center in enumerate(centers):
        center_distances[:, cluster_k] = np.sum((X - center), axis=1)
    center_distances **= (1/(fuzz-1))
    U = np.zeros((N, k))
    for cluster_k in range(k):
        U[:, cluster_k] = np.sum(center_distances, axis=1) / \
            center_distances[:, cluster_k]
    return U


# @njit(parallel=True)
def recompute_cluster_centers(X: np.ndarray, U: np.ndarray, fuzz):
    k = U.shape[1]
    centers = np.zeros((k, X.shape[1]))
    for cluster_k in prange(k):
        centers[cluster_k] = np.sum(
            X * U[:, cluster_k][:, np.newaxis]**fuzz, axis=0) / np.sum(U[:, cluster_k]**fuzz, axis=0)
    return centers


class FuzzyCMeans:
    def __init__(self, n_clusters=8, fuzz=2, init: Literal['random', 'kmeans++'] = 'kmeans++', max_iter=300):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.fuzz = fuzz

    def fit(self, X: np.ndarray):
        self.centers = fuzzy_cmeans(
            X, self.n_clusters, self.fuzz, self.init, self.max_iter)

    def predict(self, X: np.ndarray):
        clusters = NearestNeighbors(n_neighbors=1).fit(
            self.centers).kneighbors(X, return_distance=False)
        self.loss = compute_loss(clusters, self.centers, self.n_clusters)
        return clusters

    def fit_predict(self, X: np.ndarray):
        return self.fit(X).predict(X)


print()
data = np.loadtxt("Unsupervised_Learning_2024/Datasets/s3.txt")
cmeans = FuzzyCMeans(n_clusters=15, init='random')
cmeans.fit(data)
