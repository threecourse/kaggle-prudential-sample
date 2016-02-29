import sys
sys.path += [".", "utils"]
import pandas as pd
import numpy as np
from config import Config
config = Config.load()

from sklearn.externals import joblib
import subprocess
from feature_set import feature_defs, vw_inter_list
import feat_load
from util import Files
import model_util
import os

# model params
model_params_dict = {}
model_params_dict["vw1"] = {
    'l1': 1e-7,
    'passes': None,
}
model_params_dict["vw2"] = {
    'l1': 3e-8,
    'passes': 70,
}
model_params_dict["vw2_en"] = {
    'l1': 3e-8,
    'passes': 10,
}


def to_param(series):
    ret = {}
    ret["l1"] = float(series["l1"])
    ret["passes"] = int(series["passes"])
    if np.isnan(series["interaction"]):        
        ret["interaction"] = ""
    else:
        ret["interaction"] = series["interaction"]

    return ret
    
hopt_path = "../model/others/hopt_vw.txt"
if os.path.exists(hopt_path):
    param_list_hopt = model_util.to_param_list(hopt_path, to_param, sep="\t", sort_key="loss")
    for i in range(len(param_list_hopt)):
        model_params_dict["vw_hopt{}".format(i)] = param_list_hopt[i]


def run_model(ms, i_fold):

    model = ModelVW(ms.name(), i_fold)

    prms = model_params_dict[ms.model_params]
    
    if not prms.has_key("interaction"):
        prms["interaction"] = vw_inter_list[ms.feature_set]

    model.set_params(prms)
    model.set_data(ms.feature_set, i_fold)  # special

    model.train()

    pred = model.predict()
    train_pred = model.predict_train()

    model.dump()
    model.dump_pred(pred, "pred.pkl")

    return pred, train_pred

def run_model_hopt(name, _prms, feature_set_name, i_fold):
    # NOTICE: vwtxt file remains

    model = ModelVW(name, i_fold)
    prms = dict(_prms)

    model.set_params(prms)
    model.set_data(feature_set_name, i_fold)

    model.train()

    pred = model.predict()

    return pred, model
    

class ModelVW:
    # TODO: deleting text after running

    def __init__(self, model_set_name, i_fold):
        self.model_set_name = model_set_name
        self.i_fold = i_fold

    def set_params(self, prms):
        self.prms = prms
        if self.prms["passes"] is None:
            self.passes = 100
            self.early_terminate = ""
        else:
            self.passes = self.prms["passes"]
            self.early_terminate = "--early_terminate 9999"

    def set_data(self, feature_set, i_fold, create_txt=True):
        feature_list = feature_defs[feature_set]
        feature_values_with_tags = feat_load.load_feature_data_list(i_fold, feature_list)

        id_df = feat_load.load_id_df(i_fold)
        self.labels = id_df["label"].values
        self.is_tr = (id_df["id_tr"] >= 0).values
        self.is_te = (id_df["id_te"] >= 0).values
        self.labels_tr = self.labels[self.is_tr]
        self.labels_te = self.labels[self.is_te]
        self.train_len = len(self.labels_tr)
        self.all_len = len(self.labels)

        if create_txt:
            Files.mkdir(config.get_model_folder(self.model_set_name, self.i_fold))
            txt = self.create_vw_txt(self.labels, feature_values_with_tags)
            path = config.get_model_path(self.model_set_name, "vw_train.txt", self.i_fold)
            txt.to_csv(path, index=False)

    def train(self):
        self.vw_run_train()

    def predict(self):
        self.vw_run_test()
        values = self.load_pred()
        print "vw rmse", np.sqrt(((self.labels_te - values) ** 2).mean())
        # raise Exception
        return values

    def predict_train(self):
        self.vw_run_test()
        values = self.load_pred_train()
        return values

    def dump(self):
        pass

    def dump_pred(self, pred, name):
        folder = config.get_model_folder(self.model_set_name, self.i_fold)
        Files.mkdir(folder)
        path = config.get_model_path(self.model_set_name, name, self.i_fold)
        joblib.dump(pred, path)

    # private -----
    def vw_run_train(self):
        command = ("vw -d vw_train.txt -c -k -P {} --holdout_after {} --passes {} -f vw.mdl {} --l1 {} {}"
                   .format(self.train_len, self.train_len, self.passes,
                           self.prms["interaction"], self.prms["l1"], self.early_terminate))
        print command
        folder = config.get_model_folder(self.model_set_name, self.i_fold)
        subprocess.call(command, shell=True, cwd=folder)

    def vw_run_test(self):
        command = "vw -d vw_train.txt -t -i vw.mdl -p vw_pred.txt"
        print command
        folder = config.get_model_folder(self.model_set_name, self.i_fold)
        subprocess.call(command, shell=True, cwd=folder)

    def load_pred(self):
        return self.load_all()[self.is_te]

    def load_pred_train(self):
        return self.load_all()[self.is_tr]

    def load_all(self):
        "return type: np 1d array"
        path = config.get_model_path(self.model_set_name, "vw_pred.txt", self.i_fold)
        f = open(path)
        lines = f.readlines()
        values = [float(line.rstrip()) for line in lines]
        return np.array(values)

    def create_vw_txt(self, labels, feature_values_with_tags):
        n = len(labels)
        txt = pd.Series([""] * n)

        vw_labels = np.array(labels).astype(float)
        txt += pd.Series(vw_labels).astype(str)

        for c, df in feature_values_with_tags:
            txt += self.create_vw_txt_char(c, df)
            print "char {} {} converted".format(c, df.shape)
        return txt

    def create_vw_txt_char(self, char, df):
        values = df.values
        n = values.shape[0]
        m = values.shape[1]
        ret = pd.Series([" |" + char] * n)
        to_str = np.vectorize(self.to_str)
        for i in range(m):
            ret += to_str([i] * n, values[:, i].reshape(-1))
        return ret

    def to_str(self, i, v):
        if float(v) == 0.0:
            return ""
        return " {}:{:.4f}".format(i, v)
