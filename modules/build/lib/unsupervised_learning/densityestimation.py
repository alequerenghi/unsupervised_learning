import numpy as np
from numba import prange, njit


@njit(parallel=True)
def histograms(dataset: np.ndarray):
    n = dataset.shape[0]
    IQR = np.quantile(dataset, 0.75) - np.quantile(dataset, 0.25)
    delta = 2 * IQR / np.cbrt(n)
    dataset = dataset / delta
    histo = np.zeros(int(dataset.max/delta)+1)
    for i in dataset:
        histo[int(i/delta)] += 1
    return histo / n


@njit(parallel=True)
def kde(dataset: np.ndarray):
    n = dataset.shape[0]
    h = 0.9 * min(dataset.std() ** 2, np.quantile(dataset, 0.75) -
                  np.quantile(dataset, 0.25)/1.34) * n ** (- 1/5)
    rho = np.zeros(n)
    for i in prange(n):
        kernel = 0
        for j in range(i, n):
            kernel += np.exp(-(((dataset[i] - dataset[j])/h) ** 2) / 2)
        rho[i] = kernel / np.sqrt(2*np.pi) / n / h
    return rho
