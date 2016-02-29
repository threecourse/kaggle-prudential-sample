"""
    common preprocessing
"""

import sys
sys.path += [".", "utils"]
import pandas as pd
import numpy as np
from config import Config
from sklearn.preprocessing import LabelEncoder
config = Config.load()
from sklearn.externals import joblib
from util import Files

# concat train and test
if config.test:
    train = pd.read_csv("../input/train_small.csv")
    test = pd.read_csv("../input/test_small.csv")
else:
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    
train["is_train"] = True
test["Response"] = 8
test["is_train"] = False

data = pd.concat([train, test])

"""
print "----------------"
print data.columns
print "----------------"
"""

# set label and id0
data = data.rename(columns={"Response": "label"})   # 1-8
data = data.rename(columns={"Id": "id0"})


# Label Encoding
LE_prodinfo2 = LabelEncoder().fit(data["Product_Info_2"])
data["Product_Info_2_raw"] = data["Product_Info_2"]
data["Product_Info_2"] = LE_prodinfo2.transform(data["Product_Info_2"])

data = data.reset_index(drop=True)


data2 = data.copy()

print "start preprocess"

# preprocessed df
id_df = joblib.load(config.get_feat_path("id_df.pkl", None))
data3 = id_df[["id0", "id", "id_tr", "id_te"]].merge(data2, on="id0").reset_index(drop=True)

Files.mkdir(config.get_feat_folder("common"))
data3.to_pickle(config.get_feat_path("data3.pkl", "common"))
