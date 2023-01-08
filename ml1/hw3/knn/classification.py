import numpy as np

from sklearn.neighbors import NearestNeighbors
from knn.nearest_neighbors import NearestNeighborsFinder


class KNNClassifier:
    EPS = 1e-5

    def __init__(self, n_neighbors, algorithm='my_own', metric='euclidean', weights='uniform'):
        if algorithm == 'my_own':
            finder = NearestNeighborsFinder(n_neighbors=n_neighbors, metric=metric)
        elif algorithm in ('brute', 'ball_tree', 'kd_tree',):
            finder = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, metric=metric)
        else:
            raise ValueError("Algorithm is not supported", metric)

        if weights not in ('uniform', 'distance'):
            raise ValueError("Weighted algorithm is not supported", weights)

        self._finder = finder
        self._weights = weights

    def fit(self, X, y=None):
        self._finder.fit(X)
        self._labels = np.asarray(y)
        return self

    def _predict_precomputed(self, indices, distances):
        y_nearest = np.take(self._labels, indices.astype('int'))
        if self._weights == 'uniform':
            return np.apply_along_axis(lambda y: np.bincount(y).argmax(),1, y_nearest)
        inv_weights = 1/(self.EPS + distances)
        distinct_labels = np.array(list(set(self._labels)))
        weighted_scores = ((y_nearest[:, :, np.newaxis] == distinct_labels) * inv_weights[:, :, np.newaxis]).sum(axis=1)
        predictions = distinct_labels[weighted_scores.argmax(axis=1)]
        return predictions

    def kneighbors(self, X, return_distance=False):
        return self._finder.kneighbors(X, return_distance=return_distance)

    def predict(self, X):
        distances, indices = self.kneighbors(X, return_distance=True)
        return self._predict_precomputed(indices, distances)


class BatchedKNNClassifier(KNNClassifier):
    '''
    Нам нужен этот класс, потому что мы хотим поддержку обработки батчами
    в том числе для классов поиска соседей из sklearn
    '''

    def __init__(self, n_neighbors, algorithm='my_own', metric='euclidean', weights='uniform', batch_size=None):
        KNNClassifier.__init__(
            self,
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            weights=weights,
            metric=metric,
        )
        self._batch_size = batch_size

    def kneighbors(self, X, return_distance=False):
        if self._batch_size is None or self._batch_size >= X.shape[0]:
            return super().kneighbors(X, return_distance=return_distance)
        tmp_size = self._batch_size - (X.shape[0] - X.shape[0] // self._batch_size * self._batch_size)
        batches = np.vsplit(X, np.arange(0, X.shape[0], self._batch_size))[1:]
        if tmp_size < self._batch_size:
            batches[-1] = np.concatenate((batches[-1], np.zeros((tmp_size, X.shape[1]))))

        res = np.array([KNNClassifier.kneighbors(self, y, return_distance=return_distance) for y in batches])

        if return_distance:
            res = np.concatenate(res, axis=1)
            return res[0][:X.shape[0]], res[1][:X.shape[0]]
        res = np.concatenate(res, axis=0)[:X.shape[0]]
        return res
        

    def set_batch_size(self, batch_size):
        self._batch_size = batch_size