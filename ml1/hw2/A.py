import numpy as np


def get_nonzero_diag_product(X):
    d = np.diagonal(X)
    d = d[d != 0]
    if d.size == 0:
        return None
    return d.prod()
