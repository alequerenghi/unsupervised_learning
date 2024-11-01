import numpy as np


class Pca():

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, dataset):
        if self.n_components is None:
            self.n_components = dataset.shape[1]
        # Store dataset and variables to center data
        dataset = np.array(dataset)
        self.mean = np.mean(dataset, axis=0)
        self.stdv = np.std(dataset, axis=0)
        self.x = (dataset - self.mean) / self.stdv
        # Extract eigenvalues and eigenvectors and sort them
        s, u = np.linalg.eigh(np.cov(self.x, rowvar=False))
        self.s = s[-self.n_components:][::-1]
        self.u = u[:, -self.n_components:][:, ::-1]

    def transform(self, x):
        x = (x - self.mean) / self.stdv
        return x @ self.u

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(self.x)
