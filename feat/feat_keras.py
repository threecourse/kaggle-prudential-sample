"""
    feature for keras
"""

import sys
sys.path += [".", "utils"]
import pandas as pd
import numpy as np
from config import Config
config = Config.load()
from feat_util import FUtil
import feat_common
import feat_load

def create_others_keras(_data):
    "cont, dcr, MKdummy"

    df = _data.copy()
    df.columns = feat_common.replaced_columns(df.columns)
    df.index = df["id0"]
    df = df.fillna(df.mean())

    ret = []
    ret.append(("PI4", df[["PI4"]]))
    ret.append(("Ins_Age", df[["Ins_Age"]]))
    ret.append(("build", df[["Ht", "Wt", "BMI"]]))
    ret.append(("EIcont", df[["EI1", "EI4", "EI6"]]))
    ret.append(("FHcont", df[["FH2", "FH3", "FH4", "FH5"]]))
    ret.append(("IHcont", df[["IH5"]]))
    ret.append(("MHdcr", df[feat_common.dcr_variables]))
    ret.append(("MKdummy", df[feat_common.dummy_variables]))

    return ret
    
def create_isnan(_data):
    "is null or not"
    
    df = _data.copy()
    df.columns = feat_common.replaced_columns(df.columns)
    df.index = df["id0"]
    
    columns = feat_common.hasnan_variables
    df = df[columns].copy()
    for c in columns:
        df[c] = np.where(df[c].isnull(), 1.0, 0.0)
      
    ret = []
    ret.append(("isnan", df))

    return ret

def create_category1b(_data):
    "category1, majority is removed"

    df = _data.copy()
    df.columns = feat_common.replaced_columns(df.columns)
    df.index = df["id0"]
    
    lst = [
      ("cat1PIb", feat_common.cat1PI_variables),
      ("cat1EIb", feat_common.cat1EI_variables),
      ("cat1IIb", feat_common.cat1II_variables),
      ("cat1IHb", feat_common.cat1IH_variables),
      ("cat1FHb", feat_common.cat1FH_variables),
      ("cat1MHb", feat_common.cat1MH_variables),
    ]

    ret = []
    for name, cols in lst:
        df2 = df[cols].copy()
        columns = []
        for c in cols:
            se_order = feat_common.replacer_df(df2[c])["order"]
            df2["{}_PO".format(c)] = se_order.reindex(df2[c]).values
            columns += ["{}_PO".format(c)]
        df2 = df2[columns]
        df2 = df2.replace({0: np.nan})
        df2 = pd.get_dummies(df2, prefix=df2.columns, columns=df2.columns, dummy_na=False)
        ret.append((name, df2))

    return ret

def create_category2b(_data):
    "category2, threshold less than 10"

    df = _data.copy()
    df.columns = feat_common.replaced_columns(df.columns)
    df.index = df["id0"]
    
    lst = [
      ("PI2b", ["PI2"]),
      ("PI3b", ["PI3"]),
      ("EI2b", ["EI2"]),
      ("II3b", ["II3"]),
      ("MH2b", ["MH2"]),
    ]

    ret = []
    for name, cols in lst:
        df2 = df[cols].copy()
        df2 = feat_common.replace_rarevalues(df2, cols, 15, 9999)
        df2 = pd.get_dummies(df2, prefix=df2.columns, columns=df2.columns, dummy_na=False)
        ret.append((name, df2))

    return ret

if __name__ == "__main__":
    print "start feat_keras"

    # each cv folds sample
    for i_fold in config.folds:
        data3 = pd.read_pickle(config.get_feat_path("data3.pkl", "common"))
        
        id_df = feat_load.load_id_df(i_fold)        
        data3 = data3[data3["id0"].isin(id_df["id0"])].copy()
        
        FUtil.write_features(data3, create_others_keras, config, i_fold)
        FUtil.write_features(data3, create_isnan, config, i_fold)
        FUtil.write_features(data3, create_category1b, config, i_fold)
        FUtil.write_features(data3, create_category2b, config, i_fold)
        
