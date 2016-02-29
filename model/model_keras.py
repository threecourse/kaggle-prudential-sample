import sys
sys.path += [".", "utils"]
import numpy as np
import pandas as pd
np.random.seed(1337)  # for reproducibility

"""
import theano
theano.config.floatX = 'float32'
import theano.sandbox.cuda
theano.sandbox.cuda.use("gpu0")
print theano.config.floatX
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adadelta
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from sklearn.externals import joblib
from config import Config
config = Config.load()
from keras.regularizers import l2

import feat_load
from feature_set import feature_defs
import model_util
import os
from util import Files

model_params_dict = {}
model_params_dict["keras_best"] = {
    "adadelta_eps": 6.5e-08,
    "adadelta_lr": 0.25,
    "adadelta_rho_m": 0.2,
    "decay": 0.003,
    "dropout1": 0.2,
    "dropout2": 0.4,
    "h1": 450.0,
    "h2": 200.0,
    "nb_epochs": 22,
}

def to_param(series):
    ret = {}
    ret["adadelta_eps"] = float(series["adadelta_eps"])
    ret["adadelta_lr"] = float(series["adadelta_lr"])
    ret["adadelta_rho_m"] = float(series["adadelta_rho_m"])
    ret["decay"] = float(series["decay"])
    ret["dropout1"] = float(series["dropout1"])
    ret["dropout2"] = float(series["dropout2"])
    ret["h1"] = int(series["h1"])
    ret["h2"] = int(series["h2"])
    ret["nb_epochs"] = int(series["n"]) # considering hyperopt
    return ret  

hopt_path = "../model/others/hopt_keras.txt"
if os.path.exists(hopt_path):
    param_list_hopt = model_util.to_param_list(hopt_path, to_param, sep="\t", sort_key="loss")
    for i in range(len(param_list_hopt)):
        model_params_dict["keras_hopt{}".format(i)] = param_list_hopt[i]

def run_model(ms, i_fold):
    np.random.seed(71)  # for reproducibility

    model = ModelKeras(ms.name(), i_fold)

    prms = model_params_dict[ms.model_params]

    labels_tr, labels_te = feat_load.load_labels(i_fold)
    data_tr, data_te = feat_load.load_feature_data_splited(i_fold, feature_defs[ms.feature_set], scaling=True)

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

    model = ModelKeras(name, i_fold)

    labels_tr, labels_te = feat_load.load_labels(i_fold)
    data_tr, data_te = feat_load.load_feature_data_splited(i_fold, feature_defs[feature_set_name])

    model.set_params(prms)
    model.set_data(labels_tr, labels_te, data_tr, data_te)
    model.train()

    pred = model.predict()
    
    return pred, model


class ModelKeras:

    def __init__(self, model_set_name, i_fold):
        self.model_set_name = model_set_name
        self.i_fold = i_fold

    def set_params(self, prms):
        self.prms = prms

        self.nb_epochs = self.prms["nb_epochs"]
        if self.nb_epochs is None:
            self.nb_epochs = 200
            self.earlystopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2)
            self.callbacks = [self.earlystopping]
        else:
            self.callbacks = []

    def set_data(self, labels_tr, labels_te, data_tr, data_te):
        print labels_tr.shape, labels_te.shape, data_tr.shape, data_te.shape
        self.labels_tr = labels_tr
        self.labels_te = labels_te
        self.data_tr = data_tr
        self.data_te = data_te

    def train(self):

        self._construct(self.data_tr.shape[1])
        self.hist = self.model.fit(self.data_tr.values, self.labels_tr, nb_epoch=self.nb_epochs, batch_size=32, verbose=2,
                                   validation_data=(self.data_te.values, self.labels_te), callbacks=self.callbacks)
        
    def predict(self):
        return self.model.predict(self.data_te.values, batch_size=128, verbose=2).reshape(-1)

    def predict_train(self):
        return self.model.predict(self.data_tr.values, batch_size=128, verbose=2).reshape(-1)

    def dump_model(self):
        pass

    def dump_pred(self, pred, name):
        folder = config.get_model_folder(self.model_set_name, self.i_fold)
        Files.mkdir(folder)
        path = config.get_model_path(self.model_set_name, name, self.i_fold)
        joblib.dump(pred, path)

    def _construct(self, inputShape):
        init = 'glorot_normal'
        activation = 'relu'
        loss = 'mse'  
        print "loss", loss

        layers = [self.prms["h1"], self.prms["h2"]]
        dropout = [self.prms["dropout1"], self.prms["dropout2"]]
        optimizer = Adadelta(lr=self.prms["adadelta_lr"], rho=(1.0 - self.prms["adadelta_rho_m"]),
                             epsilon=self.prms["adadelta_eps"])
        decay = self.prms["decay"]

        model = Sequential()
        for i in range(len(layers)):
            if i == 0:
                print ("Input shape: " + str(inputShape))
                print ("Adding Layer " + str(i) + ": " + str(layers[i]))
                model.add(Dense(layers[i], input_dim=inputShape, init=init, W_regularizer=l2(decay)))
            else:
                print ("Adding Layer " + str(i) + ": " + str(layers[i]))
                model.add(Dense(layers[i], init=init, W_regularizer=l2(decay)))
            print ("Adding " + activation + " layer")
            model.add(Activation(activation))
            model.add(BatchNormalization())
            if len(dropout) > i:
                print ("Adding " + str(dropout[i]) + " dropout")
                model.add(Dropout(dropout[i]))
        model.add(Dense(1, init=init))  # End in a single output node for regression style output
        # ADAM=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        model.compile(loss=loss, optimizer=optimizer)

        self.model = model
