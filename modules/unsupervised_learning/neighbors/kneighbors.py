import numpy as np
from numba import njit, prange


@njit(parallel=True)
def find_kneighbors(X: np.ndarray, Y: np.ndarray):
    distances = np.zeros((Y.shape[0], X.shape[0]))
    indices = np.zeros(distances.shape, dtype=np.int64)
    for row in prange(Y.shape[0]):
        distances[row] = np.sum((Y[row] - X)**2, axis=1) ** (1/2)
        indices[row] = np.argsort(distances[row])
        distances[row] = distances[row, indices[row]]

    return distances, indices


class NearestNeighbors:
    def __init__(self, n_neighbors=5) -> None:
        self.n_neighbors = n_neighbors

    def fit(self, X: np.ndarray):
        self.X = X
        return self

    def kneighbors(self, X: np.ndarray, return_distances=True):
        distances, indices = find_kneighbors(self.X, X)
        if return_distances:
            return distances[:, :self.n_neighbors], indices[:, :self.n_neighbors]
        else:
            return indices[:, :self.n_neighbors]
