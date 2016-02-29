import sys
sys.path += [".", "utils"]
import pandas as pd
import numpy as np
from config import Config
config = Config.load()
from feat_util import FUtil 

def create_MKsum(_data):
    "only total MK"

    df = _data.copy()
    df.index = df["id0"]
    clist = ["Medical_Keyword_{}".format(i) for i in range(1, 49)]

    df["MKsum"] = df[clist].sum(axis=1)
    df = df[["MKsum"]]
    
    ret = []
    ret.append(("MKsum", df))
    return ret

if __name__ == "__main__":
    print "start feat_MKsum"
 
    data3 = pd.read_pickle(config.get_feat_path("data3.pkl", "common"))
    FUtil.write_features(data3, create_MKsum, config, "common")
