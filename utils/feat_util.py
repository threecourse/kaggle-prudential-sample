import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
from util import Files

class Feature:
    
    def __init__(self, df, name):
        self.df = df
        self.name = name

    def dump(self, fpath):
        joblib.dump(self, fpath)

    @classmethod
    def load(cls, fpath):
        return joblib.load(fpath)

class FUtil:

    @classmethod
    def create_id_df(cls, df, is_train):
        """
        :rtype: DataFrame
        :return: dataFrame, sorted by id, 
                 columns are ["label", "id0", "id", "id_tr", "id_te"]
        """

        df = df[["id0", "label"]].copy()
        df = df.reset_index(drop=True)
        is_train = np.array(is_train)

        le_tr = LabelEncoder().fit(df.id0[is_train])
        le_te = LabelEncoder().fit(df.id0[~is_train])

        df["id_tr"] = np.nan
        df["id_te"] = np.nan
        df.loc[is_train, "id_tr"] = le_tr.transform(df.id0[is_train])
        df.loc[~is_train, "id_te"] = le_te.transform(df.id0[~is_train])
        df["id"] = np.where(np.isnan(df["id_tr"]), len(le_tr.classes_) + df["id_te"], df["id_tr"])

        df = df.fillna(-1)
        df = df.sort("id")
        df = df[["label", "id0", "id", "id_tr", "id_te"]]

        return df

    @classmethod
    def write_features(cls, data, df_func, config, i_fold):
        """
        :param df_func: function, converting data into list of (name, feature dataframe)
                        index of feature dataframe should be id0
        :type df_func: DataFrame -> list of (str, Dataframe)
        """
        name_df_list = df_func(data)
        for name, df in name_df_list:
            f = Feature(df, name=name)
            print name, df.shape
            Files.mkdir(config.get_feat_folder(i_fold))
            f.dump(config.get_feat_path("{}.pkl".format(name), i_fold))
