import matplotlib.pyplot as plt


import numpy as np
from sklearn.neighbors import NearestNeighbors
from .clustering import kmeans_plus_plus
from numba import njit, prange
from typing import Literal


# @njit(parallel=True)
def fuzzy_cmeans(X: np.ndarray, k, fuzz, init, max_iter):
    N = X.shape[0]
    new_centers = np.zeros((k, X.shape[1]))
    if init == 'kmeans++':
        centers = X[kmeans_plus_plus(X, k)]
    elif type(init) == np.ndarray:
        centers = init
    else:
        centers = X[np.random.randint(X.shape[0], size=k)]
    U = np.zeros((N, k))
    for row in range(N):
        U[row, np.random.randint(k)] = 1
    # until stops moving
    for _ in range(max_iter):
        if (new_centers == centers).all():
            # if np.max(np.abs(new_centers - U)) <= 0.01:
            break
        else:
            new_centers = np.copy(centers)
            centers = recompute_cluster_centers(X, U, fuzz)
            U = recompute_fuzz(X, centers, fuzz)
    return centers


@njit(parallel=True)
def recompute_fuzz(X: np.ndarray, centers: np.ndarray, fuzz):
    N = X.shape[0]
    k = centers.shape[0]
    center_distances = np.zeros((N, k))
    for cluster_k, center in enumerate(centers):
        center_distances[:, cluster_k] = np.sum((X - center)**2, axis=1)
    center_distances = center_distances ** (1/(fuzz-1))
    U = np.zeros((N, k))
    for cluster_k in range(k):
        U[:, cluster_k] = (np.sum(center_distances, axis=1) /
                           (center_distances[:, cluster_k]) * k)
    return U / np.sum(U, axis=1).reshape(N, 1)


@njit(parallel=True)
def recompute_cluster_centers(X: np.ndarray, U: np.ndarray, fuzz):
    k = U.shape[1]
    centers = np.zeros((k, X.shape[1]))
    U = U ** fuzz
    for cluster_k in prange(k):
        UK = (U[:, cluster_k])
        centers[cluster_k] = np.sum(X * UK, axis=0) / np.sum(UK, axis=0)
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
        return self

    def predict(self, X: np.ndarray):
        distances, clusters = NearestNeighbors(
            n_neighbors=1).fit(self.centers).kneighbors(X)
        self.loss = np.sum(distances*+2)
        return clusters

    def fit_predict(self, X: np.ndarray):
        return self.fit(X).predict(X)
