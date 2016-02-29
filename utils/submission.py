import pandas as pd
import numpy as np
from config import Config
config = Config.load()
import feat_load

def make_submission(pred_class, fname):

    i_fold = None
    id_df = feat_load.load_id_df(i_fold)
    is_train = (id_df["id_tr"] >= 0)
    id0s = id_df[~is_train].id0.values

    df = pd.DataFrame(zip(id0s, pred_class), columns=["Id", "Response"])
    df["Id"] = df["Id"].astype(int)
    df["Response"] = df["Response"].astype(int)

    print "make submission"
    df.to_csv("../submission/{}.csv".format(fname), index=False)
