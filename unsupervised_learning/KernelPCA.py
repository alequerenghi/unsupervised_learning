from numba import njit, prange
import numpy as np
from typing import Literal
import matplotlib.pyplot as plt


@njit(parallel=True)
def poly_kernel(dataset, delta):
    n = dataset.shape[0]
    kernel = np.zeros((n, n))
    for i in prange(n):
        for j in range(i, n):
            kernel[i, j] = (dataset[i] @ dataset[j] + 1) ** delta
            if i-j:
                kernel[j, i] = kernel[i, j]
    return kernel


@njit(parallel=True)
def gauss_kernel(dataset, sigma):
    n = dataset.shape[0]
    kernel = np.zeros((n, n))
    for i in prange(n):
        for j in range(i, n):
            kernel[i, j] = np.exp(
                dataset[i]**2 @ dataset[j]**2 / (2 * sigma ** 2))
            if i-j:
                kernel[j, i] = kernel[i, j]
    return kernel


@njit(parallel=True)
def double_center(kernel):
    n = kernel.shape[0]
    gram = np.zeros((n, n))
    row_sum = np.zeros(n)
    for i in prange(n):
        temp = 0
        for k in range(n):
            temp += kernel[i]@kernel[k]
        row_sum[i] = temp
    for i in prange(n):
        for j in range(i, n):
            gram[i, j] = kernel[i]@kernel[j] - 1/n * row_sum[i] - \
                1/n * row_sum[j] + 1/n**2 * sum(row_sum)
            if i-j:
                gram[j, i] = gram[i, j]
    return gram


class KernelPCA():
    def __init__(self, mode: Literal['polynomial', 'gauss'], d=2, delta=2, sigma=2) -> None:
        self.mode = mode
        self.d = d
        self.delta = delta
        self.sigma = sigma

    def fit(self, dataset):
        self.dataset = dataset

    @njit
    def transform(self):
        kernel = poly_kernel() if self.mode == 'polynomial' else gauss_kernel()
        gram = double_center(kernel)
        self.s, self.u = np.linalg.eigh(gram)
        return np.sqrt(self.s[-self.d:]) * self.u[:, -self.d:]

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform()
