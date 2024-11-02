
from sklearn.neighbors import NearestNeighbors
import numpy as np
from unsupervised_learning import shortest_path


class Isomap:

    def __init__(self, alg, n_neighbors=5, n_features=None):
        self.n_neighbors = n_neighbors
        self.n_features = n_features
        self.alg = alg

    def fit(self, dataset):
        if self.n_features is None:
            self.n_features = dataset.shape[1]
        # self.mean = np.mean(dataset, axis=0)
        # self.std = np.std(dataset, axis=0)
        self.neigh = NearestNeighbors(n_neighbors=self.n_neighbors)
        self.neigh.fit(dataset)

    def transform(self, dataset):
        n = dataset.shape[0]
        # dataset = (dataset-self.mean)/self.std
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
