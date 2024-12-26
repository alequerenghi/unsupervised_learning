import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from numba import njit, prange


def _locally_linear_embedding(X, n_neighbors, n_components, reg):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(X)
    ind = nbrs.kneighbors(X, return_distance=False)[:, 1:]
    W = compute_weights(X, ind, reg)
    embedding = compute_embedding(W, n_components)
    reconstruction_error = compute_reconstruction_error(X, W, ind)
    return embedding, reconstruction_error


@njit(parallel=True)
def compute_weights(X: np.ndarray, ind: np.ndarray, reg=0.001) -> np.ndarray:
    N = X.shape[0]
    W = np.zeros(shape=(N, N), dtype=np.float64)
    for x in prange(N):
        x_neighbors = ind[x]
        C = X[x_neighbors] @ X[x_neighbors].T
        C += np.eye(C.shape[0]) * reg
        CI = np.linalg.inv(C)
        xprod = X[x_neighbors] @ X[x]
        alpha = 1 - np.sum(CI * (xprod))
        beta = CI.sum()
        l = alpha/beta
        W[x, x_neighbors] = np.sum(CI * (xprod+l), axis=1)
    return W


@njit(parallel=True)
def compute_reconstruction_error(D: np.ndarray, W: np.ndarray, indices: np.ndarray):
    N = D.shape[0]
    err = 0
    for i in prange(N):
        x_neighbors = D[indices[i]]
        x_recon = np.dot(W[i, indices[i]], x_neighbors)
        err += np.linalg.norm(D[i] - x_recon)**2
    return err


def compute_embedding(W: np.ndarray, n_components):
    M = np.eye(W.shape[0]) - W - W.T + W.T @ W
    M = coo_matrix(M)
    _, eig = eigsh(M, n_components+1, sigma=0, which='LM')
    return eig[:, 1:n_components+1]


class LLE:

    def __init__(self, n_neighbors=5, n_components=2, reg=0.001,):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.reg = reg

    def _fit_transform(self, X):
        self.embedding_, self.reconstruction_error_ = _locally_linear_embedding(
            X=X, n_neighbors=self.n_neighbors, n_components=self.n_components, reg=self.reg)

    def fit(self, X: np.ndarray):
        self._fit_transform(X)
        return self

    def fit_transform(self, X):
        self._fit_transform(X)
        return self.embedding_
