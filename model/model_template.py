import sys
sys.path += [".", "utils"]
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from config import Config
config = Config.load()
import feat_load
from feature_set import feature_defs
from util import Files

# params
model_params_dict = {}

def run_model(ms, i_fold):

    model = ModelX(ms.name(), i_fold)

    prms = model_params_dict[ms.model_params]

    labels_tr, labels_te = feat_load.load_labels(i_fold)
    data_tr, data_te = feat_load.load_feature_values_trte(i_fold, feature_defs[ms.feature_set], scaling=True)

    model.set_params(prms)
    model.set_data(labels_tr, labels_te, data_tr, data_te)
    model.train()

    pred = model.predict()
    train_pred = model.predict_train()

    model.dump_model()
    model.dump_pred(pred, "pred.pkl")

    return pred, train_pred


class ModelX:

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
        pass

    def predict(self):
        pass

    def predict_train(self):
        pass

    def dump_model(self):
        pass

    def dump_pred(self, pred, name):
        folder = config.get_model_folder(self.model_set_name, self.i_fold)
        Files.mkdir(folder)
        path = config.get_model_path(self.model_set_name, name, self.i_fold)
        joblib.dump(pred, path)
