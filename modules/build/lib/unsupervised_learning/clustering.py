import numpy as np
from sklearn.neighbors import NearestNeighbors
from numba import njit, prange, int64
from typing import Literal


@njit(parallel=True)
def mutual_information_criterion(X: np.ndarray, y: np.ndarray):
    y = y.flatten()
    mi = np.zeros(X.shape[1])
    n_rows = X.shape[0]
    n_features = max(y)+1
    # for each feature i
    for i in prange(X.shape[1]):
        n_vals = max(X[:, i])+1
        value_counts = np.zeros((n_vals, n_features))
        # count the observation wrt label and feature value
        for n in range(n_rows):
            value_counts[int(X[n, i]), int(y[n])] += 1
        temp = 0
        # compute the probabilities
        for feature in range(n_features):
            for val in range(n_vals):
                temp += value_counts[val, feature] * np.log(value_counts[val, feature] / sum(
                    value_counts[val, :]) / sum(value_counts[:, feature])*n_rows) / n_rows
        mi[i] = temp
    return mi


def kmeans(X: np.ndarray, k, init, max_iter):
    N = X.shape[0]
    neigh = NearestNeighbors(n_neighbors=1)
    new_centers = np.zeros((k, X.shape[1]))
    if init == 'kmeans++':
        centers = X[kmeans_plus_plus(X, k)]
    else:
        centers = X[np.random.randint(X.shape[0], size=k)]

    # until stops moving
    for _ in range(max_iter):
        if (new_centers == centers).all():
            break
        else:
            new_centers = np.copy(centers)
            neigh.fit(centers)
            indices = neigh.kneighbors(X, return_distance=False)
            centers = recompute_cluster_centers(X, indices, k)
    return centers


@njit(parallel=True)
def recompute_cluster_centers(X: np.ndarray, indices: np.ndarray, k):
    indices = indices.flatten()
    centers = np.zeros((k, X.shape[1]))
    for cluster_k in prange(k):
        in_cluster_k = np.where(indices == cluster_k)
        nodes = X[in_cluster_k]
        centers[cluster_k] = np.sum(nodes, axis=0) / nodes.shape[0]
    return centers


def kmedoids(X: np.ndarray, k, init, max_iter):
    N = X.shape[0]
    neigh = NearestNeighbors(n_neighbors=1)
    new_centers = np.zeros(k)
    if init == 'kmeans++':
        centers = kmeans_plus_plus(X, k)
    else:
        centers = np.random.randint(X.shape[0], size=k)

    # until stops moving
    for _ in range(max_iter):
        if (new_centers == centers).all():
            break
        else:
            new_centers = np.array(centers, copy=True)
            neigh.fit(X[centers])
            indices = neigh.kneighbors(X, return_distance=False)
            centers = recompute_medoid_centers(X, indices, k)
    return centers


@njit(parallel=True)
def recompute_medoid_centers(X: np.ndarray, indices: np.ndarray, k):
    indices = indices.flatten()
    clusters = np.zeros(k, dtype=int64)
    for cluster in prange(k):
        in_cluster = np.where(indices == cluster)[0]
        nodes = X[in_cluster] ** 2
        dist = np.zeros(len(in_cluster))
        for i in range(dist.shape[0]):
            dist[i] = np.sum(np.sqrt(abs(np.sum(nodes - nodes[i], axis=1))))
        clusters[cluster] = in_cluster[np.argmin(dist)]
    return clusters


def kmeans_plus_plus(X: np.ndarray, k: int):
    X = X ** 2
    centers = np.zeros(k, dtype=int)
    d = np.zeros(X.shape[0], dtype=int)

    centers[0] = np.random.randint(low=X.shape[0])
    # until all clusters are added
    for cluster in range(k):
        for idx, node in enumerate(X):
            closest = np.inf
            # find closest centroid
            for j in range(cluster+1):
                closest = min(
                    abs(np.sum(node - X[centers[j]])), closest)
            d[idx] = closest
        centers[cluster] = np.argmax(np.random.randint(d+1))
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
        clusters = NearestNeighbors(n_neighbors=1).fit(
            self.centers).kneighbors(X, return_distance=False)
        self.loss = self.compute_loss(clusters)
        return clusters

    def fit_predict(self, X: np.ndarray):
        return self.fit(X).predict(X)

    def compute_loss(self, clusters: np.ndarray):
        clusters = clusters ** 2
        loss = 0
        for cluster in range(self.n_clusters):
            in_cluster = np.where(clusters == cluster)[0]
            loss += np.sum(
                np.sqrt(np.abs(np.sum(clusters[in_cluster] - self.centers[cluster] ** 2, axis=1))))
        return loss


class KMedoids:
    def __init__(self, n_clusters=8, init: Literal['random', 'kmeans++'] = 'kmeans++', max_iter=300):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        return self

    def fit(self, X: np.ndarray):
        self.centers = X[kmedoids(X, self.k, self.init, self.max_iter)]
        return self

    def predict(self, X: np.ndarray):
        return NearestNeighbors(n_neighbors=1).fit(self.centers).kneighbors(X, return_distance=False)

    def fit_predict(self, X: np.ndarray):
        return self.fit(X).predict(X)
