import sys
import numpy as np


class ClusteringFeature:
    def __init__(self, data_dimensionality):
        self.N = 0
        self.LS = np.zeros(data_dimensionality)
        self.SS = 0

    def add_point(self, x):
        self.N += 1
        self.LS += np.array(x)
        self.SS += np.dot(x, x)
        return self

    def merge(self, other: 'ClusteringFeature'):
        self.N += other.N
        self.LS += other.LS
        self.SS += other.SS
        return self

    def centroid(self):
        if self.N == 0:
            return np.zeros_like(self.LS)
        return self.LS / self.N

    def radius(self):
        centroid = self.centroid()
        variance = (self.SS / self.N) - np.dot(centroid, centroid)
        # Evita numeri negativi
        variance = max(variance, 0)
        return float(np.sqrt(variance))

    def diameter(self):
        if self.N <= 1:
            return 0.0
        numerator = 2 * self.N * np.sum(self.SS) - 2 * np.dot(self.LS, self.LS)
        denominator = self.N * (self.N - 1)
        diameter_sq = numerator / denominator
        return np.sqrt(diameter_sq)

    def dist(self, other: 'ClusteringFeature') -> float:
        # Distanza euclidea tra i centroidi di due ClusteringFeature
        return float(np.linalg.norm(self.centroid() - other.centroid()))

    def __repr__(self):
        return f"ClusteringFeature(N={self.N}, LS={self.LS}, SS={self.SS:.4f})"

    def __copy__(self):
        copied = ClusteringFeature(np.size(self.LS, 0))
        copied.merge(self)
        return copied

    def __eq__(self, value: object) -> bool:
        if value is not None and isinstance(value, ClusteringFeature):
            return self.N == value.N and all(self.LS == value.LS) and self.SS == value.SS
        return False

    def __sizeof__(self) -> int:
        to_return = 0
        for element in (self.N, self.LS, self.SS):
            to_return += sys.getsizeof(element)
        return to_return
