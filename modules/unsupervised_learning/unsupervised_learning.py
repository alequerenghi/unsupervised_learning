import numpy as np
from numba import njit, prange


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
            current_node = np.int64(current_node)
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
