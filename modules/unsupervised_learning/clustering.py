import numpy as np
from sklearn.neighbors import NearestNeighbors
from numba import njit, int32, prange


@njit(parallel=True)
def mutual_information_criterion(x: np.ndarray, y: np.ndarray):
    mi = np.zeros((x.shape[1]))
    n_rows = x.shape[0]
    n_features = y.max()+1
    # for each feature i
    for i in prange(x.shape[1]):
        n_vals = x[:, i].max()+1
        value_counts = np.zeros((n_vals, n_features))
        # count the observation wrt label and feature value
        for n in range(n_rows):
            value_counts[int32(x[n, i]), int32(y[n])] += 1
        temp = 0
        # compute the probabilities
        for feature in range(n_features):
            for val in range(n_vals):
                temp += value_counts[val, feature] * np.log(
                    value_counts[val, feature] / value_counts[val, :].sum() / value_counts[:, feature].sum()*n_rows) / n_rows
        mi[i] = temp
    return mi


def kmeans(X: np.ndarray, k):
    N = X.shape[0]
    neigh = NearestNeighbors(n_neighbors=1)
    new_clusters = np.zeros((k, X.shape[1]))
    clusters = kmeans_plus_plus(X, k)

    # until stops moving
    while all(new_clusters == clusters):
        new_clusters = clusters
        neigh.fit(clusters)
        indices = neigh.kneighbors(X, return_distance=False)
        clusters = recompute_centers(X, indices, clusters)
    return clusters


def recompute_centers(X: np.ndarray, indices: np.ndarray, clusters: np.ndarray):
    for cluster_k in range(clusters.shape[0]):
        in_cluster_k = np.where(indices == cluster_k)
        nodes = X[in_cluster_k]
        clusters[cluster_k] = sum(nodes) / nodes.shape[0] 
    return clusters


def kmeans_plus_plus(X: np.ndarray, k: int):
    X = X ** 2
    clusters = np.zeros(k, X.shape[1])
    d = np.zeros(X.shape)

    clusters[0] = X[random.randint(X.shape[0])]
    # until all clusters are added
    for cluster in range(k):
        for node in X:
            closest = np.inf
            # find closest centroid
            for j in range(cluster):
                closest = min(abs(sum(X[node] - clusters[j])), closest)
            d[node] = int(closest)

        clusters[cluster] = X[np.argmax(np.random.randint(d))]
    return clusters


def kmedoids(X: np.ndarray, k, kmeans_plus_plus=True):
    N = X.shape[0]
    neigh = NearestNeighbors(n_neighbors=1)
    new_clusters = np.zeros(k)

    # initialize
    if kmeans_plus_plus:
        clusters = kmeans_plus_plus(X, k)
    else:
        clusters = np.random.randint(N, size=k)

    # until stops moving
    while all(new_clusters == clusters):
        new_clusters = clusters
        neigh.fit(clusters)
        indices = neigh.kneighbors(X, return_distance=False)
        clusters = recompute_medoids(X, indices, clusters)
    return indices


def recompute_medoids(X: np.ndarray, indices: np.ndarray, clusters: np.ndarray):
    for cluster_k in clusters:
        in_cluster_k = np.where(indices == cluster_k)
        nodes = X[in_cluster_k] ** 2
        dist = np.zeros(in_cluster_k.shape[0])
        for i in range(in_cluster_k.shape[0]):
            dist[i] = abs(sum(nodes - nodes[i]))
        clusters[cluster_k] = in_cluster_k[np.argmin(dist)]
    return clusters


def kmeans_plus_plus(X: np.ndarray, k: int):
    X = X ** 2
    clusters = np.zeros(k, X.shape[1])
    d = np.zeros(X.shape)

    clusters[0] = X[random.randint(X.shape[0])]
    # until all clusters are added
    for cluster in range(k):
        for node in X:
            closest = np.inf
            # find closest centroid
            for j in range(cluster):
                closest = min(abs(sum(X[node] - clusters[j])), closest)
            d[node] = int(closest)

        clusters[cluster] = X[np.argmax(np.random.randint(d))]
    return clusters
