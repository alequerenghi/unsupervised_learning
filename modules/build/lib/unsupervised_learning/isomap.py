
from sklearn.neighbors import NearestNeighbors
import numpy as np
from unsupervised_learning import shortest_path
from typing import Literal


class Isomap:

    def __init__(self, alg: Literal['fw', 'd'], n_neighbors=5, n_features=None):
        self.n_neighbors = n_neighbors
        self.n_features = n_features
        self.alg = alg

    def fit(self, dataset: np.ndarray):
        if self.n_features is None:
            self.n_features = dataset.shape[1]
        self.neigh = NearestNeighbors(n_neighbors=self.n_neighbors)
        self.neigh.fit(dataset)

    def transform(self, dataset: np.ndarray):
        n = dataset.shape[0]
        data, indices = self.neigh.kneighbors(dataset)
        dist = shortest_path(self.alg, data, indices)
        h = np.eye(n) - np.ones((n, n)) / n
        dist = dist ** 2
        gram = - h.dot(dist).dot(h)/2
        s, u = np.linalg.eigh(gram)
        s = s[-self.n_features:][::-1]
        u = u[:, -self.n_features:][:, ::-1]
        return np.sqrt(s) * u

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)
