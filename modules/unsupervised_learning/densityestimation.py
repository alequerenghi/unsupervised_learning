import numpy as np
from numba import prange, njit


@njit
def histograms(X: np.ndarray, bins=None):
    X = X - min(X)
    n = X.shape[0]
    if bins is None:
        IQR = np.quantile(X, 0.75) - np.quantile(X, 0.25)
        delta = 2 * IQR / np.cbrt(n)
        bins = int(max(X) / delta)+ 1
    else:
        delta = max(X)/(bins-1)
    X = X / delta
    histo = np.zeros(bins)
    for i in X:
        histo[int(i)] += 1
    return histo / n


@njit(parallel=True)
def gaussian_kernel_density(Y: np.ndarray, X: np.ndarray, bandwidth):
    # sigma = np.std(X)
    density = np.zeros(X.shape[0])
    for i in prange(X.shape[0]):
        kernel = 0
        for j in range(Y.shape[0]):
            kernel += np.exp(-(((X[i] - Y[j])/(bandwidth*2)) ** 2))
        density[i] = kernel / (np.sqrt(2*np.pi)*X.shape[0]*bandwidth)
    return density


class KernelDensity:
    def __init__(self, kernel='gaussian', bandwidth=None) -> None:
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(self, X: np.ndarray):
        self.X = X
        self.n = self.X.shape[0]
        if self.bandwidth is None:
            IQR = np.quantile(self.X, 0.75) - \
                np.quantile(self.X, 0.25)
            self.bandwidth = 0.9 * \
                min(self.X.std()**2, IQR/1.34)*self.n**(-1/5)
        return self

    def score_sample(self, X):
        # if self.kernel is 'gaussian':
        return gaussian_kernel_density(self.X, X, self.bandwidth)
