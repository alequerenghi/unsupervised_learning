from sklearn.neighbors import NearestNeighbors
import numpy as np
from typing import Literal
import heapq
from scipy.sparse import csr_matrix
from numba import njit, prange


def dijkstra(graph):
    graph = csr_matrix(graph)
    n = graph.shape[0]
    dist = np.zeros((n, n))
    dist.fill(np.inf)
    for start_node in range(n):
        pq = [(0, start_node)]
        dist[start_node, start_node] = 0
        while pq:
            current_distance, current_node = heapq.heappop(pq)
            if current_distance > dist[start_node, current_node]:
                continue
            # _, cols = graph[current_node].nonzero()
            # for neighbor in cols:
            for neighbor, weight in zip(graph[current_node].indices, graph[current_node].data):
                distance = weight + current_distance
                if distance < dist[start_node, neighbor]:
                    dist[start_node, neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))
    for i in range(n):
        for j in range(i, n):
            small = min(dist[i, j], dist[j, i])
            dist[i, j] = small
            dist[j, i] = small
    return dist


def floyd_warshall(graph):
    # Build the distance matrix with Floyd-Warshall
    n = np.size(graph, axis=0)
    d_matrix = np.zeros((n, n))
    d_matrix.fill(np.inf)
    for i in range(n):
        for j in range(n):
            # if nonzero then write in the distance matrix
            if graph[i, j]:
                d_matrix[i, j] = graph[i, j]
                d_matrix[j, i] = graph[i, j]
    for i in range(n):
        d_matrix[i, i] = 0
    for k in range(n):
        for i in range(n):
            for j in range(n):
                d_matrix[i, j] = min(
                    d_matrix[i, j], d_matrix[i, k] + d_matrix[k, j])
    return d_matrix


class Isomap:

    def __init__(self, n_neighbors=5, n_features=2):
        self.n_neighbors = n_neighbors
        self.n_features = n_features

    def fit(self, dataset):
        self.n = np.size(dataset, axis=0)
        # Build the graph
        neigh = NearestNeighbors(n_neighbors=self.n_neighbors)
        neigh.fit(dataset)
        self.graph = neigh.kneighbors_graph(dataset, 'distance')

    def transform(self, alg: Literal["dijkstra", "floyd_warshall"]):
        if (alg == 'dijkstra'):
            dist = dijkstra(self.graph)
        else:
            dist = floyd_warshall(self.graph)
        h = np.eye(self.n) - np.ones((self.n, self.n)) / self.n
        dist = dist ** 2
        gram = - h.dot(dist).dot(h)/2
        s, u = np.linalg.eig(gram)
        idx = np.argsort(s)[::-1]
        self.s = s[idx]
        self.u = u[:, idx]
        return np.sqrt(self.s[:self.n_features])*self.u[:, :self.n_features]

    def fit_transform(self, dataset, alg='dijkstra'):
        self.fit(dataset)
        return self.transform(alg)

    """        
    def multidimensionalscaling(dist, k=2):
            n = np.size(dist, axis=0)
            gram = np.zeros((n, n))
            col_sum = [np.sum(k**2) for k in dist]
            for i in range(n):
                for j in range(n):
                    gram[i, j=-1/2*dist[i,j]**2+1/2*(1/n*(col_sum[j]+col_sum[i])-1/n**2 *np.sum(col_sum))
            s, u = np.linalg.eig(gram)
            idx = np.argsort(s)[::-1]
            s = s[idx]
            u = u[idx]
            return u[:,:k] * np.sqrt(s[:k])
    
    """


"""@njit(parallel=True)
def dijkstra_par(data, indptr):
    n = indptr.shape[0]-1
    dist = np.zeros((n, n))
    dist.fill(np.inf)
    for start_node in prange(n):
        pq = [(0, start_node)]
        dist[start_node, start_node] = 0
        while pq:
            current_distance, current_node = heapq.heappop(pq)
            if current_distance > dist[start_node, current_node]:
                continue
            # _, cols = graph[current_node].nonzero()
            # for neighbor in cols:
            for neighbor, weight in zip(graph[current_node].indices, graph[current_node].data):
                distance = weight + current_distance
                if distance < dist[start_node, neighbor]:
                    dist[start_node, neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))
    for i in range(n):
        for j in range(i, n):
            small = min(dist[i, j], dist[j, i])
            dist[i, j] = small
            dist[j, i] = small
    return dist
"""
