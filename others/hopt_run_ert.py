# -*- coding: utf-8 -*-
import sys
sys.path += [".", "utils", "model", "others"]

import numpy as np
import pandas as pd
from hopt_runner import HyperOptRunner
from hyperopt import hp
from model_ert import run_model_hopt
import model_util
from util import Files
from config import Config
config = Config.load()

class HyperOptRunnerERT(HyperOptRunner):

    def init2(self):
        Files.mkdir("../model/others")

        self.fpath = "../model/others/hopt_ert.txt"
        if config.test:
            self.max_evals = 1
        else:            
            self.max_evals = 50
            
        self.space = {
            'n_estimators': hp.quniform("n_estimators", 10, 600, 10),
            'max_features': hp.quniform("max_features", 0.05, 1.0, 0.05),
            'n_jobs': -1,
            'random_state': 71
        }

        self.i_folds = [("B", 0)]

        self.output_items = []
        self.output_items += ["loss", "rsme"]
        self.output_items += ["loss{}".format(i) for i, i_fold in enumerate(self.i_folds)]
        self.output_items += ["rsme{}".format(i) for i, i_fold in enumerate(self.i_folds)]        
       
    def calculate_loss(self, params):
        prms = dict(params)
        print prms

        losses = []
        rsmes = []

        for i_fold in self.i_folds:

            pred, model = run_model_hopt("ert_hopt", prms, "fnn1", i_fold)
            ev = model_util.evaluate_label_cdf(model.labels_te, pred.reshape(-1), model.labels_tr)
            rsme = np.sqrt(((model.labels_te - pred) ** 2).mean())

            print "decode cdf -- ", ev, "---------"
            loss = -ev
            losses.append(loss)
            rsmes.append(rsme)

        ret = {}
        ret["loss"] = np.mean(losses)
        ret["rsme"] = np.mean(rsmes)

        for i, i_fold in enumerate(self.i_folds):
            ret["loss{}".format(i)] = losses[i]
            ret["rsme{}".format(i)] = rsmes[i]

        return ret

if __name__ == "__main__":
    hopt_runner = HyperOptRunnerERT()
    hopt_runner.run()
