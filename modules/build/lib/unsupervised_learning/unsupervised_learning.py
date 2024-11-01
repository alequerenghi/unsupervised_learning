from numba import njit, prange, int32
import numpy as np


def mixGauss(means, gammas, n):
    """
    Parameters:
    means: matrix/list of float of dimension n_classes x dim_data
        Means of the Gaussian functions
    gammas: array/list of float of dimension n_classes
        Standard deviation of the Gaussian functinos
    n: int
        Number of points for each class
    """
    means = np.array(means)
    gammas = np.array(gammas)

    dim = np.shape(means)[1]
    num_classes = gammas.size

    data = np.full(fill_value=np.inf, shape=(n*num_classes, dim))
    labels = np.zeros(n*num_classes)

    for i, _ in enumerate(gammas):
        data[i*n:(i+1)*n] = np.random.multivariate_normal(means[i],
                                                          np.eye(dim)*gammas[i]**2, n)
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


def build_kernel(dataset, kernel, degree, gamma):
    match kernel:
        case 'poly':
            return poly_kernel(dataset, degree, gamma)
        case 'gauss':
            return gauss_kernel(dataset, gamma)
        case _:
            return linear_kernel(dataset)


@njit(parallel=True)
def poly_kernel(dataset, degree, gamma):
    n = dataset.shape[0]
    kernel = np.zeros((n, n))
    for i in prange(n):
        for j in range(i, n):
            kernel[i, j] = (dataset[i] @ dataset[j] + gamma) ** degree
            if i-j:
                kernel[j, i] = kernel[i, j]
    return kernel


@njit(parallel=True)
def linear_kernel(dataset):
    n = dataset.shape[0]
    kernel = np.zeros((n, n))
    for i in prange(n):
        for j in range(i, n):
            kernel[i, j] = dataset[i] @ dataset[j]
            if i-j:
                kernel[j, i] = kernel[i, j]
    return kernel


@njit(parallel=True)
def gauss_kernel(dataset, gamma):
    n = dataset.shape[0]
    kernel = np.zeros((n, n))
    for i in prange(n):
        for j in range(i, n):
            kernel[i, j] = np.exp(sum(
                (dataset[i] - dataset[j])**2) / (2 * gamma ** 2))
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


def shortest_path(dataset, alg, indices, indptr, graph):
    match alg:
        case 'fw':
            return floyd_warshall(graph)
        case _:
            return dijkstra(dataset, indices, indptr)


@njit(parallel=True)
def dijkstra(data, indices, indptr):
    n = indptr.shape[0]-1
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
            for i in range(indptr[current_node], indptr[current_node+1]):
                weight = data[i]
                neighbor = indices[i]
                distance = weight + current_distance
                if distance < dist[start_node, neighbor]:
                    dist[start_node, neighbor] = distance
                    pq.append((distance, neighbor))
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
