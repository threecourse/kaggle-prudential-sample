import sys
sys.path += [".", "utils"]

import pandas as pd
import numpy as np
from config import Config
from feat_util import FUtil
config = Config.load()
from sklearn.externals import joblib
from util import Files

print "start create id"

# concat train and test

if config.test:
    train = pd.read_csv("../input/train_small.csv")
    test = pd.read_csv("../input/test_small.csv")
else:
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    
train["is_train"] = True
test["Response"] = 8  # dummy
test["is_train"] = False

# assuming id0 is unique
data = pd.concat([train, test])
data = data.rename(columns={"Response": "label"})   # 1-8
data = data.rename(columns={"Id": "id0"})

data = data.reset_index(drop=True)
data2 = data.copy()

data2_train = data2[data2.is_train].copy()


# None
id_df = FUtil.create_id_df(data2, data2["is_train"])

Files.mkdir(config.get_feat_folder(None))
joblib.dump(id_df, config.get_feat_path("id_df.pkl", None))

# not None 
for i_fold in config.folds:

    if i_fold is not None:
        char, i = i_fold
        kf = config.create_cvfold(data2_train["label"], char)

        is_train_cv = np.empty(len(data2_train))
        is_train_cv[kf[i][0]] = True
        is_train_cv[kf[i][1]] = False
        is_train_cv = is_train_cv.astype(bool)
    
        id_df = FUtil.create_id_df(data2_train, is_train_cv)
        
        Files.mkdir(config.get_feat_folder(i_fold))
        joblib.dump(id_df, config.get_feat_path("id_df.pkl", i_fold))
    
    
