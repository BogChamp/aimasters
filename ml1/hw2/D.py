import numpy as np


def get_max_after_zero(x):
    inds = np.where(x[:-1] == 0)[0]
    if len(inds) == 0:
        return None
    return np.max(np.take(x[1:], inds))
