import numpy as np
import lightgbm as lgb

def MSLE(pred, true):
    metric_name = 'MSLE'
    actuals = true.get_label() if isinstance(true, lgb.Dataset) else true
    pred = np.clip(pred, 0, None)
    res = np.log1p(actuals) - np.log1p(pred)
    cond_less = res < 0
    res = res ** 2
    res[cond_less] = res[cond_less] * 1.2
    return metric_name, res.mean(), False

def first_grad(predt, y):
    cond = (y - predt) < 0
    grad = 2 * (predt - y)
    grad[cond] *= 1.2
    return grad / len(predt)

def second_grad(predt, y):
    cond = (y - predt) < 0
    hes = np.zeros_like(predt)
    hes[cond] = 2.4
    hes[~cond] = 2
    return hes

def my_obj(predt, dmat):
    y = dmat.get_label() if isinstance(dmat, lgb.Dataset) else dmat
    grad = first_grad(predt, y)
    hess = second_grad(predt, y)
    return grad, hess