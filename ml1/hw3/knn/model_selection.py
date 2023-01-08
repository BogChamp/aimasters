from collections import defaultdict

import numpy as np

from sklearn.model_selection import KFold, BaseCrossValidator
from sklearn.metrics import accuracy_score

from knn.classification import BatchedKNNClassifier


def knn_cross_val_score(X, y, k_list, scoring, cv=None, **kwargs):
    y = np.asarray(y)

    if scoring == "accuracy":
        scorer = accuracy_score
    else:
        raise ValueError("Unknown scoring metric", scoring)

    if cv is None:
        cv = KFold(n_splits=5)
    elif not isinstance(cv, BaseCrossValidator):
        raise TypeError("cv should be BaseCrossValidator instance", type(cv))

    scores = defaultdict(list)
    clf = BatchedKNNClassifier(n_neighbors=np.amax(k_list), **kwargs)

    for train, test in cv.split(X):
        distances, indices = clf.fit(X[train], y[train]).kneighbors(X[test], return_distance=True)
        for k in k_list:
            scores[k].append(scorer(y[test], clf._predict_precomputed(indices[:,:k], distances[:,:k])))

    return scores



