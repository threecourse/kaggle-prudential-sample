import sys
sys.path += [".", "utils", "model", "others"]

from scipy.optimize import minimize
import model_util
import numpy as np
import pandas as pd
from config import Config
config = Config.load()
from scipy.optimize import fmin_powell
from ml_metrics import quadratic_weighted_kappa

class Decoding:

    @classmethod
    def raw(cls, pred_tr, pred_te, labels_tr, labels_te, data_tr, data_te):
        return np.clip(np.round(pred_te), 1, 8).astype(int)

    @classmethod
    def simple_ratio(cls, pred_tr, pred_te, labels_tr, labels_te, data_tr, data_te):
        cdf = model_util.labels_to_cdf(labels_tr)
        return model_util.reg_to_class_cdf(pred_te, cdf)

    @classmethod
    def optim_value(cls, pred_tr, pred_te, labels_tr, labels_te, data_tr, data_te, verbose=0):
        "nelder-mead partition, predicted values"

        def lossfunc(ary):
            _loss = -1.0 * model_util.evaluate_class(labels_tr, cls.to_class(pred_tr, ary))
            if verbose == 1:
                print _loss
            return _loss

        init = np.arange(1.5, 8.5, 1.0)
        res = minimize(lossfunc, init, method="Nelder-Mead")
        pred_clss = cls.to_class(pred_te, res.x)
        return pred_clss

    @classmethod
    def optim_rank(cls, pred_tr, pred_te, labels_tr, labels_te, data_tr, data_te, verbose=0):
        "nelder-mead partition, for ranking"

        def lossfunc(ary):
            _loss = -1.0 * model_util.evaluate_class(labels_tr, cls.to_class(pred_tr, ary))
            if verbose == 1:
                print _loss
            return _loss

        pred_tr = pred_tr.argsort().argsort() / float(len(pred_tr))
        pred_te = pred_te.argsort().argsort() / float(len(pred_te))

        init = np.zeros(7)
        for i in range(7):
            init[i] = np.where(labels_tr <= (i + 1), 1.0, 0.0).sum() / float(len(labels_tr))
        print init

        res = minimize(lossfunc, init, method="Nelder-Mead")
        pred_clss = cls.to_class(pred_te, res.x)
        return pred_clss

    @classmethod
    def to_class(cls, pred, partition):
        pred_class = np.ones(len(pred))
        for i in range(0, 7):
            # _, 0,1,2,3,4,5,6 -> 1,2,3,4,5,6,7,8
            mask = pred > partition[i]
            pred_class[mask] = i + 2
        return pred_class

    @classmethod
    def btb(cls, pred_tr, pred_te, labels_tr, labels_te, data_tr, data_te, verbose=0):

        train_preds = np.clip(pred_tr, -0.99, 8.99)
        test_preds = np.clip(pred_te, -0.99, 8.99)
        num_classes = 8

        # train offsets
        offsets = np.ones(num_classes) * -0.5
        offset_train_preds = np.vstack((train_preds, train_preds, labels_tr))
        for j in range(num_classes):
            def train_offset(x):
                return -apply_offset(offset_train_preds, x, j)

            offsets[j] = fmin_powell(train_offset, offsets[j])

        # apply offsets to test
        data = np.vstack((test_preds, test_preds))
        for j in range(num_classes):
            data[1, data[0].astype(int) == j] = data[0, data[0].astype(int) == j] + offsets[j]

        pred_class = np.round(np.clip(data[1], 1, 8)).astype(int)
        return pred_class

    @classmethod
    def btb2(cls, pred_tr, pred_te, labels_tr, labels_te, data_tr, data_te, verbose=0):
        train_preds = np.clip(pred_tr, -0.99, 8.99)
        test_preds = np.clip(pred_te, -0.99, 8.99)
        num_classes = 8

        # train offsets
        offsets = np.array([0.1, -1, -2, -1, -0.8, 0.02, 0.8, 1])
        # offsets = np.ones(num_classes) * -0.5
        offset_train_preds = np.vstack((train_preds, train_preds, labels_tr))
        for j in range(num_classes):
            def train_offset(x):
                return -apply_offset(offset_train_preds, x, j)

            offsets[j] = fmin_powell(train_offset, offsets[j])

        # apply offsets to test
        data = np.vstack((test_preds, test_preds))
        for j in range(num_classes):
            data[1, data[0].astype(int) == j] = data[0, data[0].astype(int) == j] + offsets[j]

        pred_class = np.round(np.clip(data[1], 1, 8)).astype(int)
        return pred_class

    @classmethod
    def btb_rank(cls, pred_tr, pred_te, labels_tr, labels_te, data_tr, data_te, verbose=0):

        pred_tr = pred_tr.argsort().argsort() / float(len(pred_tr))
        pred_te = pred_te.argsort().argsort() / float(len(pred_te))

        init = np.zeros(7)
        for i in range(7):
            init[i] = np.where(labels_tr <= (i + 1), 1.0, 0.0).sum() / float(len(labels_tr))
        print init

        init2 = np.array(init)

        for j in range(7):

            def lossfunc(x):
                ary = np.array(init2)
                ary[j] = x
                _loss = -1.0 * model_util.evaluate_class(labels_tr, cls.to_class(pred_tr, ary))
                if verbose == 1:
                    print _loss
                return _loss

            init2[j] = fmin_powell(lossfunc, init2[j])

        pred_clss = cls.to_class(pred_te, init2)
        return pred_clss


def eval_wrapper(yhat, y):
    y = np.array(y)
    y = y.astype(int)
    yhat = np.array(yhat)
    yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)
    return quadratic_weighted_kappa(yhat, y)

def apply_offset(data, bin_offset, sv, scorer=eval_wrapper):
    # data has the format of pred=0, offset_pred=1, labels=2 in the first dim
    data[1, data[0].astype(int) == sv] = data[0, data[0].astype(int) == sv] + bin_offset
    score = scorer(data[1], data[2])
    return score
