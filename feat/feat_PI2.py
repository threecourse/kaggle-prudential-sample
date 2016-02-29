import sys
sys.path += [".", "utils"]
import pandas as pd
import numpy as np
from config import Config
config = Config.load()
from sklearn.preprocessing import LabelEncoder
from feat_util import FUtil 

def create_PI2cn(_data):
    df = _data.copy()
    df.index = df["id0"]

    df["PI2c"] = map(lambda x: x[0], df["Product_Info_2_raw"])
    df["PI2n"] = map(lambda x: int(x[1]), df["Product_Info_2_raw"])
    df["PI2c"] = LabelEncoder().fit_transform(df["PI2c"])

    ret = []
    ret.append(("PI2c", df[["PI2c"]]))
    ret.append(("PI2n", df[["PI2n"]]))
    return ret

if __name__ == "__main__":
    print "start feat_PI2"

    data3 = pd.read_pickle(config.get_feat_path("data3.pkl", "common"))
    FUtil.write_features(data3, create_PI2cn, config, "common")
