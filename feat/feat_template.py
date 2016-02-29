import sys
sys.path += [".", "utils"]
import pandas as pd
import numpy as np
from config import Config
config = Config.load()
from feat_util import FUtil 
import feat_load

def create_featureX(_data):
    """
    :rtype: list of (str, dataframe)
    :return: list of (name, feature dataframe)
    index of feature dataframe should be id0
    """

    df = _data.copy()
    df.index = df["id0"]
    df = df.drop(["id", "id_tr", "id_te", "id0", "label", "is_train", "Product_Info_2_raw"], axis=1)
   
    ret = []
    ret.append(("featX", df))
    
    return ret

if __name__ == "__main__":
    print "start featureX"

    # common in cv folds
    data3 = pd.read_pickle(config.get_feat_path("data3.pkl", None))
    FUtil.write_features(data3, create_featureX, config, "common")

    # --------- or --------
    # each cv folds
    for i_fold in config.folds:
        id_df = feat_load.load_id_df(i_fold)        
        data3 = data3[data3["id0"].isin(id_df["id0"])].copy()
        
        data3 = pd.read_pickle(config.get_feat_path("data3.pkl", "common"))
        FUtil.write_features(data3, create_featureX, config, i_fold)
