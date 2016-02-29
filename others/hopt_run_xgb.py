# -*- coding: utf-8 -*-
import sys
sys.path += [".", "utils", "model", "others"]

import os
import numpy as np
import pandas as pd
from hopt_runner import HyperOptRunner
from hyperopt import hp
from model_xgb import run_model_hopt
import model_util
from util import Files
from config import Config
config = Config.load()

class HyperOptRunnerXgb(HyperOptRunner):

    def init2(self):
        Files.mkdir("../model/others")

        self.fpath = "../model/others/hopt_xgb.txt"
        
        if config.test:
            self.max_evals = 1
        else:            
            self.max_evals = 50
            
        self.space = {
            'task': 'regression',
            'booster': 'gbtree',
            'objective': 'reg:linear',
            'eval_metric': "rmse",
            'eta': hp.quniform('eta', 0.001, 0.05, 0.001),
            'gamma': hp.quniform('gamma', 0.0, 1.0, 0.1),
            'min_child_weight': hp.quniform('min_child_weight', 3, 7, 1),
            'max_depth': hp.quniform('max_depth', 5, 30, 1),
            'subsample': hp.quniform('subsample', 0.7, 1, 0.1),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.4, 0.7, 0.1),
            "seed": 12345,
            'lambda': hp.loguniform("lambda", np.log(0.1), np.log(10)),
            'alpha': hp.loguniform("alpha", np.log(0.1), np.log(10)),
            'num_rounds': None,
            "silent": 1,
        }
        self.i_folds = [("B", 0)]

        self.output_items = []
        self.output_items += ["loss", "n", "rsme"]
        self.output_items += ["loss{}".format(i) for i, i_fold in enumerate(self.i_folds)]
        self.output_items += ["n{}".format(i) for i, i_fold in enumerate(self.i_folds)]
        self.output_items += ["rsme{}".format(i) for i, i_fold in enumerate(self.i_folds)]

    def calculate_loss(self, params):
        prms = dict(params)
        print prms

        losses = []
        ns = []
        rsmes = []

        for i_fold in self.i_folds:

            pred, model = run_model_hopt("xgb_hopt", prms, "f1", i_fold)

            ev = model_util.evaluate_label_cdf(model.labels_te, pred.reshape(-1), model.labels_tr)
            rsme = np.sqrt(((model.labels_te - pred) ** 2).mean())
            n = model.model.best_iteration

            print "decode cdf -- ", ev, "---------"
            loss = -ev
            losses.append(loss)
            ns.append(n)
            rsmes.append(rsme)

        ret = {}
        ret["loss"] = np.mean(losses)
        ret["n"] = np.mean(ns)
        ret["rsme"] = np.mean(rsmes)

        for i, i_fold in enumerate(self.i_folds):
            ret["loss{}".format(i)] = losses[i]
            ret["n{}".format(i)] = ns[i]
            ret["rsme{}".format(i)] = rsmes[i]

        return ret

if __name__ == "__main__":

    hopt_runner = HyperOptRunnerXgb()
    hopt_runner.run()
