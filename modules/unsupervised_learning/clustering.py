import numpy as np
import random
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


def kmeans(dataset, k, kmeans_plus_plus=True):
    # pick clusters at random or with kmenas++
    if kmeans_plus_plus:
        clusters = kmeans_plus_plus(dataset, k)
    else:
        clusters = []
        for i in range(k):
            clusters.append(random.randint(dataset.shape[0]))
    neigh = NearestNeighbors(n_neighbors=1)
    # contains the index of the assigned cluster
    assigned_clusters = [clusters[0]] * dataset.shape[0]
    # repeat until stops moving
    while True:
        neigh.fit(dataset[clusters])
        new_clusters = [] * clusters.shape[0]
        # assign to closest center
        data, indices = neigh.kneighbors(dataset)
        # recompute centers
        for i in range(k):
            weight = 0
            n = 0
            for j in range(indices.shape[0]):
                if indices[i] is k:
                    weight += data[i]
                    n += 1
            new_clusters[i] = weight / n
        # clusers stop movingK
        if (new_clusters != clusters).any():
            break
        clusters = new_clusters
        assigned_clusters = indices
    return assigned_clusters


def kmeans_plus_plus(dataset, k):
    # pick center at random
    clusters = []
    clusters.add(random.randint(dataset.shape[0]))
    # until all clusters are added
    for cluster in range(k):
        d = [0] * dataset.shape[0]
        # for all instances do
        for node in dataset:
            closest = 100
            # find closest centroid
            for j in range(k - cluster):
                temp = abs(dataset[node] ** 2 - dataset[cluster[j]])
                if temp < closest:
                    closest = temp
            d[node] = closest
        # use ptobability to add new node
        current = 0
        new_cluster = 0
        for node in range(dataset.shape[0]):
            temp = random.randint(int(d[node]))
            if temp > current:
                current = temp
                new_cluster = node
        clusters.append(new_cluster)
    return clusters


"""
def kmedoids(dataset, clusters, assigned):
    for cluster in clusters:
        previously_assigned  = []
        for node in range(dataset.shape[0]):
            if assigned[node] == cluster:
                previously_assigned.append(dataset[node])
        np.median(previously_assigned)
        """
