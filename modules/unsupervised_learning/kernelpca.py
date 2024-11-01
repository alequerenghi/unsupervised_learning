from typing import Literal
from unsupervised_learning import build_kernel, double_center
import numpy as np


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
