import sys
sys.path += [".", "utils"]

import pandas as pd
import numpy as np
from config import Config
config = Config.load()
import xgboost as xgb
from sklearn.externals import joblib
from StringIO import StringIO
import feat_load
from feature_set import feature_defs
import model_util
import os
from util import Files
import re

# from sklearn.preprocessing import StandardScaler, Imputer

# model params
model_params_dict = {}

model_params_dict["xgbtree1_3"] = {
    'objective': 'reg:linear',
    'eta': 0.025,  # 0.05
    'max_depth': 8,
    'subsample': 0.85,
    'colsample_bytree': 0.8,
    "silent": 1,
    "seed": 12345,
    'eval_metric': "rmse",
    # "min_child_weight": 1,
    'num_rounds': 370,
}

model_params_dict["xgbtree1_3en"] = dict(model_params_dict["xgbtree1_3"])
model_params_dict["xgbtree1_3en"]["num_rounds"] = 200

def to_param(series):
    ret = {}
    ret["objective"] = series["objective"]
    ret["silent"] = int(series["silent"])
    ret["seed"] = int(series["seed"])
    ret["eval_metric"] = series["eval_metric"]
    ret["num_rounds"] = int(series["n"]) # NOTICE: considering hyperopt
    ret["alpha"] = float(series["alpha"])
    ret["colsample_bytree"] = float(series["colsample_bytree"])
    ret["eta"] = float(series["eta"])
    ret["gamma"] = float(series["gamma"])
    ret["lambda"] = float(series["lambda"])
    ret["max_depth"] = int(series["max_depth"])
    ret["min_child_weight"] = int(series["min_child_weight"])
    ret["subsample"] = float(series["subsample"])
    return ret

xgb_param_str = StringIO(
"""objective   silent  seed    eval_metric n  alpha   colsample_bytree    eta gamma   lambda  max_depth   min_child_weight    subsample
reg:linear  1   12345   rmse    602    9.698   0.4 0.013   0.90    8.157   17  6   0.8
reg:linear  1   12345   rmse    538    1.423   0.4 0.014   0.40    0.284   12  6   0.8
reg:linear  1   12345   rmse    439    0.438   0.4 0.024   0.00    0.136   9   7   0.8
reg:linear  1   12345   rmse    1343    0.246   0.5 0.005   0.80    8.026   13  6   0.9
reg:linear  1   12345   rmse    1634    2.698   0.5 0.005   0.60    3.512   10  3   0.9
""")

xgb_param_list = model_util.to_param_list(xgb_param_str, to_param, sep="\s+")

model_params_dict["xgb_bbbb1"] = xgb_param_list[0]
model_params_dict["xgb_bbbb2"] = xgb_param_list[1]
model_params_dict["xgb_bbbb3"] = xgb_param_list[2]
model_params_dict["xgb_bbbb4"] = xgb_param_list[3]
model_params_dict["xgb_bbbb5"] = xgb_param_list[4]

hopt_path = "../model/others/hopt_xgb.txt"
if os.path.exists(hopt_path):
    xgb_param_list_hopt = model_util.to_param_list(hopt_path, to_param, sep="\t", sort_key="loss")
    for i in range(len(xgb_param_list_hopt)):
        model_params_dict["xgb_hopt{}".format(i)] = xgb_param_list_hopt[i]

def run_model(ms, i_fold):

    model = ModelXgb(ms.name(), i_fold)

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

    model = ModelXgb(name, i_fold)

    labels_tr, labels_te = feat_load.load_labels(i_fold)
    data_tr, data_te = feat_load.load_feature_data_splited(i_fold, feature_defs[feature_set_name])

    model.set_params(prms)
    model.set_data(labels_tr, labels_te, data_tr, data_te)
    model.train()

    pred = model.predict()
    
    return pred, model
    

class ModelXgb:

    def __init__(self, model_set_name, i_fold):
        self.model_set_name = model_set_name
        self.i_fold = i_fold

    def set_params(self, prms):
        self.prms = prms

        self.num_rounds = prms["num_rounds"]
        self.early_stopping_rounds = None
        if self.num_rounds is None:
            self.early_stopping_rounds = 5
            self.num_rounds = 20000

    def set_data(self, labels_tr, labels_te, data_tr, data_te):
        print "start constructing xgb matrix"
        self.labels_tr = labels_tr
        self.labels_te = labels_te
        
        # TODO update xgboost
        columns = [re.sub("[_\.]", "", s) for s in data_tr.columns]
        # print columns
        
        self.dtrain = xgb.DMatrix(data_tr.values, label=labels_tr, missing=np.nan, feature_names=columns)
        self.dvalid = xgb.DMatrix(data_te.values, label=labels_te, missing=np.nan, feature_names=columns)

    def train(self):
        print "start xgb"
        watchlist = [(self.dtrain, 'train'), (self.dvalid, 'eval')]
        self.model = xgb.train(self.prms, self.dtrain, num_boost_round=self.num_rounds, evals=watchlist,
                               early_stopping_rounds=self.early_stopping_rounds)
        # print self.model.get_fscore()
        

    def predict(self):
        return self.model.predict(self.dvalid)

    def predict_train(self):
        return self.model.predict(self.dtrain)

    def dump_model(self):
        folder = config.get_model_folder(self.model_set_name, self.i_fold)
        Files.mkdir(folder)
        name = "model"
        path = config.get_model_path(self.model_set_name, name, self.i_fold)
        self.model.save_model(path)

    def dump_pred(self, pred, name):
        folder = config.get_model_folder(self.model_set_name, self.i_fold)
        Files.mkdir(folder)
        path = config.get_model_path(self.model_set_name, name, self.i_fold)
        joblib.dump(pred, path)
