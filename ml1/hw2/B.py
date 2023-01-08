import numpy as np
from numpy.lib.stride_tricks import as_strided


def calc_expectations(h, w, X, Q):
    sub_shape = (h, w)
    conv_shape = (Q.shape[0] + h - 1, Q.shape[1] + w - 1)
    Q_conv = np.zeros(conv_shape)
    Q_conv[h - 1:, w - 1:] = Q
    view_shape = tuple(np.subtract(conv_shape, sub_shape) + 1) + sub_shape
    view = as_strided(Q_conv, view_shape, Q_conv.strides * 2)
    view = view.reshape((-1,) + sub_shape)
    Q_new = np.sum(view.reshape(view.shape[0], -1), axis=1).reshape(Q.shape)
    return X * Q_new
