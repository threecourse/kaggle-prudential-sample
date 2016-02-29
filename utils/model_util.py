import pandas as pd
import numpy as np
from ml_metrics import quadratic_weighted_kappa

# ---- parameter ----
def to_param_list(filepath_or_buffer, to_param, sep="\t", sort_key=None):
    param_df = pd.read_csv(filepath_or_buffer, sep=sep, index_col=False, engine="python")
    if sort_key is not None:
        # NOTICE: the smallest is the top
        param_df = param_df.sort(sort_key) 
    param_list = [to_param(se) for i, se in param_df.iterrows()]
    return param_list
    
# ---- evaluation ----
def evaluate_label_cdf(labels_te, pred, labels_tr):
    pred_class = reg_to_class_label(pred, labels_tr)
    return quadratic_weighted_kappa(pred_class, labels_te, 1, 8)

def evaluate_class(labels_te, pred_class):
    return quadratic_weighted_kappa(pred_class, labels_te, 1, 8)

def logloss(act, pred):
    epsilon = 1e-15
    pred = np.maximum(epsilon, pred)
    pred = np.minimum(1-epsilon, pred)
    ll = (act * np.log(pred) + (1.0 - act) * np.log(1.0 - pred)).sum()
    ll *= -1.0 / len(act)
    return ll

# ---- conversion ----

def reg_to_class_label(pred, labels_tr):
    cdf = labels_to_cdf(labels_tr)
    return reg_to_class_cdf(pred, cdf)

# encode continious value to class with cdf
def reg_to_class_cdf(pred, cdf):
    num = pred.shape[0]
    classes = 8

    assert(len(cdf) == classes + 1)
    assert(cdf[0] == 0.0)
    assert(cdf[classes] == 1.0)

    output = np.asarray([classes]*num, dtype=int)
    rank = pred.argsort()
    for r in range(classes - 1):  # 0 to 6
        frm = int(num*cdf[r])
        to = int(num*cdf[r+1])
        output[rank[frm:to]] = r + 1
        # print frm, to
    return output

def classprob_to_reg(pred_classprob):
    "return np.array(N * 1)"
    classes = 8
    factors = np.arange(1, classes+1).reshape(1, 8)
    return (pred_classprob * factors).sum(axis=1).reshape(-1)

def reg_to_rankratio(pred):
    N = pred.shape[0]
    rank = pred.argsort()  # order?
    output = np.zeros(N)
    output[rank] = np.arange(N) / (N * 1.0)
    return output

def labels_to_cdf(labels):
    hist = np.bincount(labels.astype(int))
    return np.cumsum(hist) / float(sum(hist))

# simple decoding
def reg_to_class_round(pred):
    pred = np.round(pred)
    pred = np.minimum(8, pred)
    pred = np.maximum(1, pred)
    return pred
