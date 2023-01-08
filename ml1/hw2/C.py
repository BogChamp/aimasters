import numpy as np


def replace_nan_to_means(X):
    Y = X.copy()
    Y_means = np.nanmean(Y, axis=0)
    inds = np.where(np.isnan(Y))
    Y[inds] = np.take(Y_means, inds[1])
    return Y
