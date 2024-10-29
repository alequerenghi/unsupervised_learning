from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import heapq
from numba import njit, prange
from typing import Literal
import numpy as np


def mixGauss(means, sigmas, n):
    """
    Parameters:
    means: matrix/list of float of dimension n_classes x dim_data
        Means of the Gaussian functions
    sigmas: array/list of float of dimension n_classes
        Standard deviation of the Gaussian functinos
    n: int
        Number of points for each class
    """
    means = np.array(means)
    sigmas = np.array(sigmas)

    dim = np.shape(means)[1]
    num_classes = sigmas.size

    data = np.full(fill_value=np.inf, shape=(n*num_classes, dim))
    labels = np.zeros(n*num_classes)

    for i, _ in enumerate(sigmas):
        data[i*n:(i+1)*n] = np.random.multivariate_normal(means[i],
                                                          np.eye(dim)*sigmas[i]**2, n)
        labels[i*n:(i+1)*n] = i

    return data, labels


def label_noise(p, labels):
    """
    Parameters:
    p: float
        Percentage of labels to flip
    labels: array of int of dimension n_points
        Array containing label indexes
    """
    n = np.shape(labels)[0]
    noisylabels = np.copy(np.squeeze(labels))
    n_flips = int(np.floor(n*p))
    idx_flip = np.random.choice(n, n_flips, False)
    noisylabels[idx_flip] = 1-noisylabels[idx_flip]
    return noisylabels


def swiss_roll(n):
    """
    Parameters:
    n: int
        Numbers of points to generate
    """

    data = np.zeros((n, 3))
    phi = np.random.uniform(1.5*np.pi, 4.5*np.pi, n)
    psi = np.random.uniform(0, 10, n)
    data[:, 0] = phi*np.cos(phi)
    data[:, 1] = phi*np.sin(phi)
    data[:, 2] = psi
    return data


@njit(parallel=True)
def poly_kernel(dataset, delta):
    n = dataset.shape[0]
    kernel = np.zeros((n, n))
    for i in prange(n):
        for j in range(i, n):
            kernel[i, j] = (dataset[i] @ dataset[j] + 1) ** delta
            if i-j:
                kernel[j, i] = kernel[i, j]
    return kernel


@njit(parallel=True)
def gauss_kernel(dataset, sigma):
    n = dataset.shape[0]
    kernel = np.zeros((n, n))
    for i in prange(n):
        for j in range(i, n):
            kernel[i, j] = np.exp(
                dataset[i]**2 @ dataset[j]**2 / (2 * sigma ** 2))
            if i-j:
                kernel[j, i] = kernel[i, j]
    return kernel


@njit(parallel=True)
def double_center(kernel):
    n = kernel.shape[0]
    gram = np.zeros((n, n))
    row_sum = np.zeros(n)
    for i in prange(n):
        temp = 0
        for k in range(n):
            temp += kernel[i]@kernel[k]
        row_sum[i] = temp
    for i in prange(n):
        for j in range(i, n):
            gram[i, j] = kernel[i]@kernel[j] - 1/n * row_sum[i] - \
                1/n * row_sum[j] + 1/n**2 * sum(row_sum)
            if i-j:
                gram[j, i] = gram[i, j]
    return gram


class KernelPCA():
    def __init__(self, mode: Literal['polynomial', 'gauss'], d=2, delta=2, sigma=2) -> None:
        self.mode = mode
        self.d = d
        self.delta = delta
        self.sigma = sigma

    def fit(self, dataset):
        self.dataset = dataset

    def transform(self):
        kernel = poly_kernel(
            self.dataset, self.delta) if self.mode == 'polynomial' else gauss_kernel(self.dataset, self.sigma)
        gram = double_center(kernel)
        self.s, self.u = np.linalg.eigh(gram)
        return np.sqrt(self.s[-self.d:]) * self.u[:, -self.d:]

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform()


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


class Pca():

    def __init__(self, d=2):
        self.d = d

    def fit(self, dataset):
        # Store dataset and variables to center data
        self.dataset = dataset
        self.mean = np.mean(self.dataset, axis=0)
        self.stdv = np.std(self.dataset, axis=0)
        self.x = (self.dataset - self.mean) / self.stdv
        # Extract eigenvalues and eigenvectors and sort them
        s, u = np.linalg.eig(np.cov(self.x, rowvar=False))
        idx = np.argsort(s)[::-1]
        self.s = s[idx]
        self.u = u[:, idx]

    def transform(self, x):
        x = (x - self.mean) / self.stdv
        return np.matmul(x, self.u[:, :self.d])

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(self.x)
