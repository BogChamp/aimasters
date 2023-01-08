import numpy as np


def euclidean_distance(x, y):
    x2 = np.sum(x**2, axis=1)
    y2 = np.sum(y**2, axis=1)
    xy = np.matmul(x, y.T)
    dists = x2[:, np.newaxis] - 2*xy + y2
    return np.sqrt(dists)


def cosine_distance(x, y):
    dots = np.dot(x, y.T)
    l2norms = np.sqrt(((x ** 2).sum(1)[:, None]) * ((y ** 2).sum(1)))
    cosine_dists = 1 - (dots / l2norms)
    return cosine_dists
