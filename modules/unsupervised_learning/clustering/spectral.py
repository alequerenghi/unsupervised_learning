from sklearn.datasets import load_iris
from typing import Literal
import numpy as np
from unsupervised_learning.neighbors import NearestNeighbors
from unsupervised_learning.clustering import KMeans


def compute_laplacian(distances: np.ndarray, indices: np.ndarray):
    N = distances.shape[0]
    S = np.zeros((N, N))
    D = np.sum(distances, axis=1)
    for row in range(N):
        for idx, col in enumerate(indices[row]):
            S[row, col] = 1
            S[col, row] = 1
        S[row, row] = -D[row]
    return -S, D


def ratio_cut(L: np.ndarray, n_clusters):
    _, U = np.linalg.eigh(L)
    return U[:, :n_clusters]


def min_cut(L: np.ndarray, D: np.ndarray, n_clusters):
    L = (D ** (-1/2)).reshape(-1, 1) * (L) * (D ** (-1/2))
    _, U = np.linalg.eigh(L)
    return U[:, :n_clusters]


class SpectralClustering:
    def __init__(self, n_clusters=8, n_neighbors=5, max_iter=300, alg: Literal['ratio_cut', 'min_cut'] = 'min_cut'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.alg = alg
        self.n_neighbors = n_neighbors

    def fit(self, X: np.ndarray):
        distances, indices = NearestNeighbors(
            self.n_neighbors).fit(X).kneighbors(X)
        L, D = compute_laplacian(distances, indices)
        if self.alg == 'ratio_cut':
            affinity_matrix = ratio_cut(L, self.n_clusters)
        else:
            affinity_matrix = min_cut(L, D, self.n_clusters)
        self.centers = KMeans(self.n_clusters, max_iter=self.max_iter).fit(
            affinity_matrix).centers
        self.affinity_matrix = affinity_matrix
        return self

    def fit_predict(self, X: np.ndarray):
        self.fit(X)
        distances, indices = NearestNeighbors(
            1).fit(self.centers).kneighbors(self.affinity_matrix)
        self.loss = np.sum(distances)
        return indices
