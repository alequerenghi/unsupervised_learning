import numpy as np


def kneighbors(X: np.ndarray, Y: np.ndarray):
    neigh = np.zeros((Y.shape[0], X.shape[0]))


class NearestNeighbors:
    def __init__(self) -> None:
        pass

    def fit(self, X: np.ndarray, n_neighbors):
        self.X = X
        self.n_neighbors = n_neighbors

    def kneighbors(self, X: np.ndarray):
        pass
