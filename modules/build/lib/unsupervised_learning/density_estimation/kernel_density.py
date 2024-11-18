from numba import njit, prange
import numpy as np


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
