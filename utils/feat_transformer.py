# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class PercentileBin:
    def transform(self, df, i_fold):
        ret_ary = []

        ary = df.values
        ret_cols = []

        percentiles = [5, 10, 25, 50, 75, 90, 95]
        
        for i in range(ary.shape[1]):
            v = ary[:, i]
            for p in percentiles:
                p_val = np.percentile(v, p)
                ret_ary.append((v < p_val).reshape(-1, 1))
                ret_cols.append("{}_p{}".format(df.columns[i], p))

        ret_ary = np.hstack(ret_ary).astype(int)
        return pd.DataFrame(ret_ary, index=df.index, columns=ret_cols)


class MeanAbs:
    def transform(self, df, i_fold):
        ret_ary = []
        ret_cols = []
        ary = df.values

        for i in range(ary.shape[1]):
            v = ary[:, i]
            vv = np.abs(v - np.mean(v))
            ret_ary.append(vv.reshape(-1, 1))
            ret_cols.append("{}_meanabs".format(df.columns[i]))

        ret_ary = np.hstack(ret_ary)
        return pd.DataFrame(ret_ary, index=df.index, columns=ret_cols)

class OneHot:
    def transform(self, df, i_fold):
        return pd.get_dummies(df, columns=df.columns)

"""
class Filter:
    def __init__(self, mask):
        self.mask = mask

    def transform(self, ary, i_fold):
        return ary[self.mask]
        
class Multiply:
    def __init__(self, mul_features):
        self.mul_features = mul_features

    def transform(self, ary, i_fold):
        ret = []
        for fname in self.mul_features:
            mul_ary = load_feature(fname, i_fold).values
            for i in range(ary.shape[1]):
                for j in range(mul_ary.shape[1]):
                    ret.append((ary[:, i] * mul_ary[:, j]).reshape(-1, 1))
        ret = np.hstack(ret)
        print "multiplied", ret.shape
        return ret
"""