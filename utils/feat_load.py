import numpy as np
import pandas as pd
from feat_util import Feature
from config import Config
config = Config.load()
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import os

# TODO dealing with column names
# TODO daatframe or np.array

# id df
def load_id_df(i_fold):
    id_df = joblib.load(config.get_feat_path("id_df.pkl", i_fold))
    return id_df

# load labels
def load_labels(i_fold):
    id_df = load_id_df(i_fold)
    labels, is_train = id_df.label.values, (id_df["id_tr"] >= 0).values
    labels_tr, labels_te = labels[is_train], labels[~is_train]
    return labels_tr, labels_te

# ---- load feature data ----

def load_feature_data_splited(i_fold, feature_list, scaling=False):
    """
    :type feature_list: list of (str, list of str, list of Transformer)
    :rtype: (DataFrame, DataFrame)
    :return: feature dataframe, train/validation
    """
    id_df = load_id_df(i_fold)
    is_train = (id_df["id_tr"] >= 0).values
    id0_train = id_df[is_train]["id0"].values
    id0_test = id_df[~is_train]["id0"].values
    
    feature_data = load_feature_data(i_fold, feature_list, scaling=scaling)
    return feature_data.reindex(id0_train), feature_data.reindex(id0_test)

def load_feature_data(i_fold, feature_list, scaling=False):
    """
    :type feature_list: list of (str, list of str, list of Transformer)
    :rtype: DataFrame
    :return: feature dataframe
    """
    feature_data_list = load_feature_data_list(i_fold, feature_list)
    feature_data = pd.concat([df for tag, df in feature_data_list], axis=1)
    if scaling:
        feature_data = scale_df(feature_data)
        
    print "loaded feature shape:", feature_data.shape
    return feature_data

def load_feature_data_list_splited(i_fold, feature_list, scaling=False):
    """
    :type feature_list: list of (str, list of str, list of Transformer)
    :rtype: (list of (str, DataFrame), list of (str, DataFrame))
    :return: list of (tag, feature dataframe), train/validation
    """
    id_df = load_id_df(i_fold)
    
    is_train = (id_df["id_tr"] >= 0).values
    id0_train = id_df[is_train]["id0"].values
    id0_test = id_df[~is_train]["id0"].values
    
    feature_data_list = load_feature_data_list(i_fold, feature_list)

    if scaling:
        feature_data_list = [(tag, scale_df(df)) for tag, df in feature_data_list]

    tr = [(tag, df.reindex(id0_train)) for tag, df in feature_data_list]
    te = [(tag, df.reindex(id0_test)) for tag, df in feature_data_list]

    return tr, te

def load_feature_data_list(i_fold, feature_list):
    """
    :type feature_list: list of (str, list of str, list of Transformer)
    :rtype: list of (str, DataFrame)
    :return: list of (tag, feature dataframe)
    """

    id_df = load_id_df(i_fold)
    
    # load into dictionary
    fdict = {}
    for tag, feat_names, transformers in feature_list:
        for fname in feat_names:
            f = load_feature(fname, i_fold)
            
            # reindexed by id_df
            df = f.df.reindex(id_df["id0"])

            for transformer in transformers:
                df = transformer.transform(df, i_fold)

            if tag in fdict:
                fdict[tag] = pd.concat([fdict[tag], df], axis=1)
            else:
                fdict[tag] = df

    # convert into list
    tags = sorted(fdict.keys())
    return [(tag, fdict[tag]) for tag in tags]

def scale_df(df):
    return pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns, index=df.index)

# ---- load feature ----

def load_feature(name, i_fold):
    """
    :rtype: Feature
    """
    
    if isinstance(name, tuple):
        # feature for stacking
        model_name, fname = name
        return load_pred_feature(model_name, fname, i_fold)
    else:
        if os.path.exists(config.get_feat_path("{}.pkl".format(name), "common")):
            return Feature.load(config.get_feat_path("{}.pkl".format(name), "common"))
        else:
            return Feature.load(config.get_feat_path("{}.pkl".format(name), i_fold))

# load feature for stacking
def load_pred_feature(model_name, fname, i_fold):
    # NOTICE: pred are written as np.array
    
    def load_preds(model_name, fname):
        id0 = []
    
        # folds
        for i in config.folds:
            id_df = load_id_df(i)
            id0 += [id_df[id_df["id_te"] >= 0].id0.values]
    
        preds = []
        for i_fold_pred in config.folds:
            values = joblib.load(config.get_model_path(model_name, fname, i_fold_pred))
            if values.ndim == 1:
                values = values.reshape(-1, 1)
            preds += [values]
        preds = np.vstack(preds)
            
        index_id0 = np.concatenate(id0)
        columns = ["{}_{}_{}".format(model_name, fname, i) for i in range(preds.shape[1])]
        pred_df = pd.DataFrame(preds, index=index_id0, columns=columns)
    
        return pred_df    
    
    blend_df = load_preds(model_name, fname)
    return Feature(blend_df, name="{}_{}".format(model_name, fname))

