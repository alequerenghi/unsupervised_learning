from typing import Literal
from sklearn.neighbors import NearestNeighbors
import numpy as np
from unsupervised_learning import shortest_path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


class Isomap:

    def __init__(self, alg: Literal["dijkstra", "floyd_warshall"], n_neighbors=5, n_features=None):
        self.n_neighbors = n_neighbors
        self.n_features = n_features
        self.alg = alg

    def fit(self, dataset):
        if self.n_features is None:
            self.n_features = dataset.shape[1]
        # self.mean = np.mean(dataset, axis=0)
        # self.std = np.std(dataset, axis=0)
        self.neigh = NearestNeighbors(n_neighbors=self.n_neighbors)
        self.neigh.fit(dataset)

    def transform(self, dataset):
        n = dataset.shape[0]
        # dataset = (dataset-self.mean)/self.std
        data, indices = self.neigh.kneighbors(dataset)
        dist = shortest_path(self.alg, data, indices)
        h = np.eye(n) - np.ones((n, n)) / n
        dist = dist ** 2
        gram = - h.dot(dist).dot(h)/2
        s, u = np.linalg.eigh(gram)
        s = s[-self.n_features:][::-1]
        u = u[:, -self.n_features:][:, ::-1]
        return np.sqrt(s) * u

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)


beans = pd.read_excel(
    'Unsupervised_Learning_2024/Datasets/Dry_Bean_Dataset.xlsx')
samples = beans.sample(1000)
y = np.array(samples['Class'])
encoder = OrdinalEncoder()
encoder.fit(y.reshape(-1, 1))
y = encoder.transform(y.reshape(-1, 1))
x = np.array(samples.drop('Class', axis=1))
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
x_train = (x_train-x_train.mean())/x_train.std()
x_test = (x_test-x_test.mean())/x_test.std()
isomap = Isomap('dijkstra', n_neighbors=20)
x_iso = isomap.fit_transform(x_train)

score = []
for i in range(x_iso.shape[1]):
    model = LogisticRegression(max_iter=1000)
    model.fit(x_iso[:, :i+1], y_train.ravel())
    x_test_pca = isomap.transform(x_test)
    x_test_pca = x_test_pca[:, :i+1]
    y_pred = model.predict(x_test_pca)
    print(f"{i+1}: {model.score(x_test_pca, y_test)}")
    score.append(model.score(x_test_pca, y_test))
plt.plot(score, "o-")
plt.show()
print(f"The maximum values of the accuaracy score is reached with {
      np.argmax(score)} PCs and it is equal to {np.max(score)}")
