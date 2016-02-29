import sys
sys.path += [".", "utils"]
import pandas as pd
import numpy as np
from config import Config
config = Config.load()
from sklearn.externals import joblib
from sklearn.ensemble import ExtraTreesRegressor

import feat_load
from feature_set import feature_defs
import model_util
import os
from util import Files

# model params
model_params_dict = {}
model_params_dict["ert1"] = {
    'n_estimators': 10,
    'max_features': "auto",
    'n_jobs': -1,
    'random_state': 71,
}

model_params_dict["ert2"] = {
    'n_estimators': 430,
    'max_features': 0.3,
    'n_jobs': -1,
    'random_state': 71,
}


def to_param(series):
    ret = {}
    ret["n_estimators"] = int(series["n_estimators"])
    ret["max_features"] = float(series["max_features"])
    ret["n_jobs"] = -1
    ret["random_state"] = 71
    return ret
    
hopt_path = "../model/others/hopt_ert.txt"
if os.path.exists(hopt_path):
    param_list_hopt = model_util.to_param_list(hopt_path, to_param, sep="\t", sort_key="loss")
    for i in range(len(param_list_hopt)):
        model_params_dict["ert_hopt{}".format(i)] = param_list_hopt[i]    

def run_model(ms, i_fold):

    model = ModelERT(ms.name(), i_fold)

    prms = model_params_dict[ms.model_params]
    labels_tr, labels_te = feat_load.load_labels(i_fold)
    data_tr, data_te = feat_load.load_feature_data_splited(i_fold, feature_defs[ms.feature_set])

    model.set_params(prms)
    model.set_data(labels_tr, labels_te, data_tr, data_te)

    model.train()

    pred = model.predict()
    train_pred = model.predict_train()

    model.dump_model()
    model.dump_pred(pred, "pred.pkl")
    model.dump_pred(train_pred, "train_pred.pkl")

    return pred, train_pred


def run_model_hopt(name, prms, feature_set_name, i_fold):

    model = ModelERT(name, i_fold)

    labels_tr, labels_te = feat_load.load_labels(i_fold)
    data_tr, data_te = feat_load.load_feature_data_splited(i_fold, feature_defs[feature_set_name])

    model.set_params(prms)
    model.set_data(labels_tr, labels_te, data_tr, data_te)
    model.train()

    pred = model.predict()

    return pred, model


class ModelERT:

    def __init__(self, model_set_name, i_fold):
        self.model_set_name = model_set_name
        self.i_fold = i_fold

    def set_params(self, prms):
        self.prms = prms

    def set_data(self, labels_tr, labels_te, data_tr, data_te):
        self.labels_tr = labels_tr
        self.labels_te = labels_te
        self.data_tr = data_tr
        self.data_te = data_te

    def train(self):
        print "start ert"
        self.model = ExtraTreesRegressor(n_jobs=self.prms["n_jobs"],
                                         verbose=1,
                                         random_state=self.prms["random_state"],
                                         n_estimators=int(self.prms["n_estimators"]),
                                         max_features=self.prms["max_features"])
        self.model.fit(self.data_tr.values, self.labels_tr)

    def predict(self):
        return self.model.predict(self.data_te.values)

    def predict_train(self):
        return self.model.predict(self.data_tr.values)

    def dump_model(self):
        pass

    def dump_pred(self, pred, name):
        folder = config.get_model_folder(self.model_set_name, self.i_fold)
        Files.mkdir(folder)
        path = config.get_model_path(self.model_set_name, name, self.i_fold)
        joblib.dump(pred, path)
