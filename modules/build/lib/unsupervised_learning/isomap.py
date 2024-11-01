from typing import Literal
from sklearn.neighbors import NearestNeighbors
import numpy as np
from unsupervised_learning import shortest_path


class Isomap:

    def __init__(self, alg: Literal["dijkstra", "floyd_warshall"], n_neighbors=5, n_features=None):
        self.n_neighbors = n_neighbors
        self.n_features = n_features
        self.alg = alg

    def fit(self, dataset):
        self.n = dataset.shape[0]
        if not self.n_features:
            self.n_features = dataset.shape[1]
        self.mean = np.mean(dataset, axis=0)
        self.std = np.std(dataset, axis=0)

    def transform(self, dataset):
        dataset = (dataset-self.mean)/self.std
        neigh = NearestNeighbors(n_neighbors=self.n_neighbors)
        neigh.fit(dataset)
        # distances, indices = neigh.kneighbors(dataset)
        self.graph = neigh.kneighbors_graph(dataset)
        dist = shortest_path(self.graph.data, self.alg,
                             self.graph.indices, self.graph.indptr, self.graph)
        h = np.eye(self.n) - np.ones((self.n, self.n)) / self.n
        dist = dist ** 2
        gram = - h.dot(dist).dot(h)/2
        s, u = np.linalg.eigh(gram)
        s = s[-self.n_features:][::-1]
        u = u[:, -self.n_features:][:, ::-1]
        return np.sqrt(s) * u

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)
