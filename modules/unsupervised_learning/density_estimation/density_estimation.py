import numpy as np
from numba import prange, njit


@njit
def histograms(X: np.ndarray, bins=0):
    X = X - min(X)
    n = X.shape[0]
    if bins is 0:
        IQR = np.quantile(X, 0.75) - np.quantile(X, 0.25)
        delta = 2 * IQR / np.cbrt(n)
        bins = int(max(X) / delta) + 1
    else:
        delta = max(X)/(bins-1)
    X = X / delta
    histo = np.zeros(bins)
    for i in X:
        histo[int(i)] += 1
    return histo / n
