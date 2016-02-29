# -*- coding: utf-8 -*-
import sys
sys.path += [".", "utils", "model", "others"]

import numpy as np
import pandas as pd
from hopt_runner import HyperOptRunner
from hyperopt import hp
from model_vw import run_model_hopt
import model_util
from util import Files
from config import Config
config = Config.load()

class HyperOptRunnerVW(HyperOptRunner):

    def init2(self):
        Files.mkdir("../model/others")

        self.fpath = "../model/others/hopt_vw.txt"
        if config.test:
            self.max_evals = 1
        else:            
            self.max_evals = 25
            
        # TODO: read passes from hyperopt result
        self.space = {
            "l1": hp.qloguniform("l1", np.log(1e-9), np.log(1e-6), 1e-9),
            'interaction': "",
            "passes": 15
        }

        self.i_folds = [("B", 0)]

        self.output_items = []
        self.output_items += ["loss", "rsme"]
        self.output_items += ["loss{}".format(i) for i, i_fold in enumerate(self.i_folds)]
        self.output_items += ["rsme{}".format(i) for i, i_fold in enumerate(self.i_folds)]        

        self.columns = sorted(self.space.keys())
       
        
    def calculate_loss(self, params):
        prms = dict(params)
        print prms

        losses = []
        rsmes = []

        for i_fold in self.i_folds:

            pred, model = run_model_hopt("vw_hopt", prms, "fvw1", i_fold)
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
    hopt_runner = HyperOptRunnerVW()
    hopt_runner.run()
