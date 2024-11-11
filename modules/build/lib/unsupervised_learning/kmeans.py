import numpy as np
import random
from sklearn.neighbors import NearestNeighbors


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


def variables_ranking(dataset):

    return 0


"""
def kmedoids(dataset, clusters, assigned):
    for cluster in clusters:
        previously_assigned  = []
        for node in range(dataset.shape[0]):
            if assigned[node] == cluster:
                previously_assigned.append(dataset[node])
        np.median(previously_assigned)
        """
