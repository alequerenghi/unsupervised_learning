
from sklearn.neighbors import NearestNeighbors
import numpy as np
from typing import Literal
from numba import njit, prange, int32


def shortest_path(alg, data, indices):
    match alg:
        case 'fw':
            return floyd_warshall(data, indices)
        case _:
            return dijkstra(data, indices)


@njit(parallel=True)
def dijkstra(data, indices):
    n = indices.shape[0]
    n_neighbors = indices.shape[1]
    dist = np.zeros((n, n))
    dist.fill(np.inf)
    for start_node in prange(n):
        pq = [(0, start_node)]
        dist[start_node, start_node] = 0
        while pq:
            current_distance, current_node = pq.pop(0)
            current_node = int32(current_node)
            if current_distance > dist[start_node, current_node]:
                continue
            for i in range(1, n_neighbors):
                neighbor = indices[current_node, i]
                weight = data[current_node, i]
                distance = weight + current_distance
                if distance < dist[start_node, neighbor]:
                    # small = min(distance, dist[neighbor, start_node])
                    dist[start_node, neighbor] = distance  # small
                    # dist[neighbor, start_node] = small
                    pq.append((distance, neighbor))
    for i in range(n):
        for j in range(i, n):
            small = min(dist[i, j], dist[j, i])
            dist[i, j] = small
            dist[j, i] = small
    return dist


@njit(parallel=True)
def floyd_warshall(data, indices):
    # Build the distance matrix with Floyd-Warshall
    n = indices.shape[0]
    n_neighbors = indices.shape[1]
    dist = np.zeros((n, n))
    dist.fill(np.inf)
    for i in prange(n):
        for j in range(n_neighbors):
            neighbor = indices[i, j]
            dist[i, neighbor] = data[i, j]
    for k in range(n):
        for i in prange(n):
            for j in range(n):
                dist[i, j] = min(
                    dist[i, j], dist[i, k] + dist[k, j])
    return dist


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
