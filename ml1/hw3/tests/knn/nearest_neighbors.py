import numpy as np

from knn.distances import euclidean_distance, cosine_distance


def get_best_ranks(ranks, top, axis=1, return_ranks=False):
    if top >= ranks.shape[axis]:
        ind = np.argsort(ranks, axis=axis)
        ind_part = ind
    else: 
        ind = np.argpartition(ranks, top, axis=axis)
        ind = np.take(ind, np.arange(top), axis=axis)

        ranks = np.take_along_axis(ranks, ind, axis=axis)

        ind_part = np.argsort(ranks, axis=axis)
        ind = np.take_along_axis(ind, ind_part, axis=axis)

    if return_ranks:
        return np.take_along_axis(ranks, ind_part, axis=axis), ind
    return ind
    


class NearestNeighborsFinder:
    def __init__(self, n_neighbors, metric="euclidean"):
        self.n_neighbors = n_neighbors

        if metric == "euclidean":
            self._metric_func = euclidean_distance
        elif metric == "cosine":
            self._metric_func = cosine_distance
        else:
            raise ValueError("Metric is not supported", metric)
        self.metric = metric

    def fit(self, X, y=None):
        self._X = X
        return self

    def kneighbors(self, X, return_distance=False):
        distances = self._metric_func(X, self._X)

        return get_best_ranks(distances, self.n_neighbors, return_ranks=return_distance)
