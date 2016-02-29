# -*- coding: utf-8 -*-
import sys
sys.path += [".", "utils", "model", "others"]

import os
import numpy as np
import pandas as pd
from hopt_runner import HyperOptRunner
from hyperopt import hp
from model_keras import run_model_hopt
import model_util
from util import Files
from config import Config
config = Config.load()

class HyperOptRunnerKeras(HyperOptRunner):

    def init2(self):
        Files.mkdir("../model/others")

        self.fpath = "../model/others/hopt_keras.txt"
        
        if config.test:
            self.max_evals = 1
        else:            
            self.max_evals = 50
            
        self.space = {
             "adadelta_eps": hp.loguniform("adadelta_eps", np.log(1e-07), np.log(1e-05)),
             "adadelta_lr": hp.loguniform("adadelta_lr", np.log(0.1), np.log(1.0)),
             "adadelta_rho_m": hp.loguniform("adadelta_rho_m", np.log(0.01), np.log(0.1)),
             "decay": hp.loguniform("decay", np.log(0.0001), np.log(0.1)),
             "dropout1": hp.quniform('dropout1', 0.1, 0.5, 0.1),
             "dropout2": hp.quniform('dropout2', 0.1, 0.5, 0.1),
             "h1": hp.quniform('h1', 50, 500, 10),  # 450.0,
             "h2": hp.quniform('h2', 20, 250, 5),  # 200.0,
             "nb_epochs": None,  # 22,
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

            pred, model = run_model_hopt("keras_hopt", prms, "fnn1", i_fold)

            ev = model_util.evaluate_label_cdf(model.labels_te, pred.reshape(-1), model.labels_tr)
            rsme = np.sqrt(((model.labels_te - pred) ** 2).mean())
            n = len(model.hist.epoch)

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

    hopt_runner = HyperOptRunnerKeras()
    hopt_runner.run()
