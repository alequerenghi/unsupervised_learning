from numba import prange
import numpy as np


def mixGauss(means, gammas, n):
    """
    Parameters:
    means: matrix/list of float of dimension n_classes x dim_data
        Means of the Gaussian functions
    gammas: array/list of float of dimension n_classes
        Standard deviation of the Gaussian functinos
    n: int
        Number of points for each class
    """
    means = np.array(means)
    gammas = np.array(gammas)

    dim = np.shape(means)[1]
    num_classes = gammas.size

    data = np.full(fill_value=np.inf, shape=(n*num_classes, dim))
    labels = np.zeros(n*num_classes)

    for i, _ in enumerate(gammas):
        data[i*n:(i+1)*n] = np.random.multivariate_normal(means[i],
                                                          np.eye(dim)*gammas[i]**2, n)
        labels[i*n:(i+1)*n] = i

    return data, labels


def label_noise(p, labels):
    """
    Parameters:
    p: float
        Percentage of labels to flip
    labels: array of int of dimension n_points
        Array containing label indexes
    """
    n = np.shape(labels)[0]
    noisylabels = np.copy(np.squeeze(labels))
    n_flips = int(np.floor(n*p))
    idx_flip = np.random.choice(n, n_flips, False)
    noisylabels[idx_flip] = 1-noisylabels[idx_flip]
    return noisylabels


def swiss_roll(n):
    """
    Parameters:
    n: int
        Numbers of points to generate
    """

    data = np.zeros((n, 3))
    phi = np.random.uniform(1.5*np.pi, 4.5*np.pi, n)
    psi = np.random.uniform(0, 10, n)
    data[:, 0] = phi*np.cos(phi)
    data[:, 1] = phi*np.sin(phi)
    data[:, 2] = psi
    return data


def histograms(dataset):
    dataset = np.sort(dataset)
    n = dataset.shape[0]
    delta = 2 * (dataset[int(n/4*3)]-dataset[int(n/4)]) / np.cbrt(n)
    dataset = dataset / delta
    histo = [0] * (int(dataset[n-1]/delta)+1)
    for i in dataset:
        histo[int(i/delta)] += 1
    return histo


def kde(dataset):
    n = dataset.shape[0]
    sorted = np.sort(dataset)
    h = 0.9 * min(dataset.std() ** 2,
                  sorted[int(n/4*3)]-sorted[int(n/4)]/1.34) * n ** (- 1/5)
    rho = [0] * n
    for i in prange(n):
        kernel = 0
        for j in range(i, n):
            kernel += np.exp(-(((dataset[i] - dataset[j])/h) ** 2) / 2)
        rho[i] = kernel / np.sqrt(2*np.pi) / n / h
    return rho
