import sys
sys.path += [".", "utils"]
import pandas as pd
import numpy as np
from config import Config
config = Config.load()
from feat_util import FUtil 

def create_basic(_data):
    "basic features"

    df = _data.copy()
    df.index = df["id0"]
    df = df.drop(["id", "id_tr", "id_te", "id0", "label", "is_train", "Product_Info_2_raw"], axis=1)
    
    ret = []
    ret.append(("basic", df))
    
    return ret

if __name__ == "__main__":
    print "start feat_basic"

    data3 = pd.read_pickle(config.get_feat_path("data3.pkl", "common"))
    FUtil.write_features(data3, create_basic, config, "common")
