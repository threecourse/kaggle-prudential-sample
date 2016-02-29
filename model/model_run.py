import sys
sys.path += [".", "utils"]
import pandas as pd
import numpy as np
from config import Config
config = Config.load()
import feat_load

import model_util
from util import Writer, Files
import model_xgb
import os
import model_keras
import model_vw
import model_ert
# import model_keras_graph
# import model_keras_binary
# import model_xgb_classify
# import model_xgb_binary
# import model_ert_binary
# import model_svr
# import model_svr2
# import model_svr3
from optim import Decoding
import submission

def run_model(labels_tr, labels_te, ms, i_fold):

    if ms.model_type in "xgb":
        return model_xgb.run_model(ms, i_fold)
    if ms.model_type == "keras":
        return model_keras.run_model(ms, i_fold)
    if ms.model_type == "vw":
        return model_vw.run_model(ms, i_fold)
    if ms.model_type == "ert":
        return model_ert.run_model(ms, i_fold)

    """
    if ms.model_type in "xgb_binary":
        return model_xgb_binary.run_model(ms, i_fold)
    if ms.model_type == "xgb_classify":
        return model_xgb_classify.run_model_xgb_classify(ms, i_fold)
    if ms.model_type == "keras_binary":
        return model_keras_binary.run_model(ms, i_fold)
    if ms.model_type == "keras_graph":
        return model_keras_graph.run_model_keras_graph(ms, i_fold)
    if ms.model_type == "ert_binary":
        return model_ert_binary.run_model(ms, i_fold)
    if ms.model_type == "svr":
        return model_svr.run_model_svr(ms, i_fold)
    if ms.model_type == "svr2":
        return model_svr2.run_model_svr2(ms, i_fold)
    if ms.model_type == "svr3":
        return model_svr3.run_model_svr3(ms, i_fold)
    """    

def run_model_all(model_settings, create_submission=False):

    fname_score_df = "../submission/score_df_run_all.csv"
    if not os.path.exists(fname_score_df):
        Writer.create_empty(fname_score_df)
        Writer.append_line_list(fname_score_df, ["model_type", "model_params", "feature_set", "i_fold", "score", "version", "rsme"])

    for ms in model_settings:
        for i in config.folds:
            labels_tr, labels_te = feat_load.load_labels(i)
            pred, pred_train = run_model(labels_tr, labels_te, ms, i)

            if i is not None:
                # scores
                ev = model_util.evaluate_label_cdf(labels_te, pred, labels_tr)
                rsme = np.sqrt(((labels_te - pred) ** 2).mean())
                print "decode cdf", ev, i
                print "rsme", rsme
                Writer.append_line_list(fname_score_df, [ms.model_type, ms.model_params, ms.feature_set,
                                                       i, ev, config.version, rsme])
                
                if create_submission:
                    pred_class = Decoding.optim_rank(pred_train, pred, labels_tr, None, None, None, verbose=0)
                    ev_optim = model_util.evaluate_class(labels_te, pred_class)
                    print "decode optim", ev_optim, i

            if i is None:
                
                if create_submission:
                    pred_class = Decoding.optim_rank(pred_train, pred, labels_tr, None, None, None, verbose=0)                   
                    print "make submission"
                    submission.make_submission(pred_class, "_".join([config.version, ms.name(), "optim_rank"]))

        # show scores
        score_df = pd.read_csv(fname_score_df, sep="\t")
        g = score_df.groupby(["model_type", "model_params", "feature_set", "version"])
        print g["score"].agg([np.mean, np.std]).reset_index()
   