import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd


class IntrinsicDimension():

    def __init__(self) -> None:
        pass

    def estimate(self, dataset: np.ndarray):
        n = dataset.shape[0]
        neigh = NearestNeighbors(n_neighbors=3)
        neigh.fit(dataset)
        data, _ = neigh.kneighbors(dataset)
        mu = [0] * n
        for i in range(n):
            mu[i] = data[i, 2]/data[i, 1]

        return n / sum(np.log(mu))
