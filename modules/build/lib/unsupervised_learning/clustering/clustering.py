import numpy as np
from sklearn.neighbors import NearestNeighbors


def kmeans_plus_plus(X: np.ndarray, k: int):
    centers = np.zeros(k, dtype=int)
    d = np.zeros(X.shape[0], dtype=int)

    centers[0] = np.random.randint(low=X.shape[0])
    # until all clusters are added
    for cluster in range(k):
        distances = NearestNeighbors(n_neighbors=1).fit(
            X[centers[:cluster+1]]).kneighbors(X)[0].flatten()
        distances.astype(int)
        centers[cluster] = np.argmax(np.random.randint(distances**2+1))
    return centers
