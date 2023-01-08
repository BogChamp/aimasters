import numpy as np


def encode_rle(x):
    diffs = x[1:] != x[:-1]
    inds = np.append(np.where(diffs), len(x) - 1)
    z = np.diff(np.append(-1, inds))
    return (x[inds], z)
