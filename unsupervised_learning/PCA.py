import numpy as np


class Pca():

    def __init__(self, d=2):
        self.d = d

    def fit(self, dataset):
        # Store dataset and variables to center data
        self.dataset = dataset
        self.mean = np.mean(self.dataset, axis=0)
        self.stdv = np.std(self.dataset, axis=0)
        self.x = (self.dataset - self.mean) / self.stdv
        # Extract eigenvalues and eigenvectors and sort them
        s, u = np.linalg.eig(np.cov(self.x, rowvar=False))
        idx = np.argsort(s)[::-1]
        self.s = s[idx]
        self.u = u[:, idx]

    def transform(self, x):
        x = (x - self.mean) / self.stdv
        return np.matmul(x, self.u[:, :self.d])

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(self.x)
