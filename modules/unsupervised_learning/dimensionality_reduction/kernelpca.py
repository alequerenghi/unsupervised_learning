from typing import Literal
# from unsupervised_learning import build_kernel, double_center
import numpy as np
from numba import njit, prange


def build_kernel(dataset, kernel, degree, gamma):
    match kernel:
        case 'poly':
            return poly_kernel(dataset, degree, gamma)
        case 'gauss':
            return gauss_kernel(dataset, gamma)
        case _:
            return linear_kernel(dataset)


@njit(parallel=True)
def poly_kernel(dataset, degree, gamma):
    n = dataset.shape[0]
    kernel = np.zeros((n, n))
    for i in prange(n):
        for j in range(i, n):
            kernel[i, j] = (dataset[i] @ dataset[j] + gamma) ** degree
            if i-j:
                kernel[j, i] = kernel[i, j]
    return kernel


@njit(parallel=True)
def linear_kernel(dataset):
    n = dataset.shape[0]
    kernel = np.zeros((n, n))
    for i in prange(n):
        for j in range(i, n):
            kernel[i, j] = dataset[i] @ dataset[j]
            if i-j:
                kernel[j, i] = kernel[i, j]
    return kernel


@njit(parallel=True)
def gauss_kernel(dataset, gamma):
    n = dataset.shape[0]
    kernel = np.zeros((n, n))
    for i in prange(n):
        for j in range(i, n):
            kernel[i, j] = np.exp(-sum(
                (dataset[i] - dataset[j])**2) / (2 * gamma ** 2))
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

    def __init__(self, mode: Literal['linear', 'poly', 'gauss'], n_components=2, degree=3, gamma=None) -> None:
        self.mode = mode
        self.d = n_components
        self.degree = degree
        self.gamma = gamma

    def fit(self, dataset):
        self.dataset = dataset
        if not self.gamma:
            self.gamma = 1 / self.dataset.shape[0]

    def transform(self):
        kernel = build_kernel(self.dataset, self.mode, self.degree, self.gamma)
        gram = double_center(kernel)
        self.s, self.u = np.linalg.eigh(gram)
        return np.sqrt(self.s[-self.d:]) * self.u[:, -self.d:]

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform()
