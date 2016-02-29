"""
    common variables and functions
"""

import sys
sys.path += [".", "utils"]
import pandas as pd
import numpy as np
import re

# variables by attribute

cont_variables = ["PI4", "Ins_Age", "Ht", "Wt", "BMI", "EI1", "EI4", "EI6", "IH5", "FH2", "FH3", "FH4", "FH5"]
dcr_variables = ["MH1", "MH10", "MH15", "MH24", "MH32"]
cat1PI_variables = ['PI1', 'PI5', 'PI6', 'PI7']
cat1EI_variables = ['EI3', 'EI5']
cat1II_variables = ['II1', 'II2', 'II4', 'II5', 'II6', 'II7']
cat1IH_variables = ['IH1', 'IH2', 'IH3', 'IH4', 'IH7', 'IH8', 'IH9']
cat1FH_variables = ['FH1']
cat1MH_variables = ['MH3', 'MH4', 'MH5', 'MH6', 'MH7', 'MH8', 'MH9', 'MH11', 'MH12', 'MH13',
                    'MH14', 'MH16', 'MH17', 'MH18', 'MH19', 'MH20', 'MH21', 'MH22', 'MH23', 'MH25',
                    'MH26', 'MH27', 'MH28', 'MH29', 'MH30', 'MH31', 'MH33', 'MH34', 'MH35', 'MH36',
                    'MH37', 'MH38', 'MH39', 'MH40', 'MH41']
cat2_variables = ["PI2", "PI3", "EI2", "II3", "MH2"]
dummy_variables = ["MK{}".format(i) for i in range(1, 49)]

hasnan_variables = ["EI1", "EI4", "EI6", "IH5", "FH2", "FH3", "FH4", "FH5"] + dcr_variables
categorical_variables = (cat1PI_variables + cat1EI_variables + cat1II_variables +
                         cat1IH_variables + cat1FH_variables + cat1MH_variables +
                         cat2_variables)

# variables by group
PIcols = ["Ins_Age", "Ht", "Wt", "BMI"] + ["PI{}".format(i) for i in range(1, 8)]
EIcols = ["EI1", "EI2", 'EI3', "EI4", 'EI5', "EI6"]
IIcols = ["II{}".format(i) for i in range(1, 8)]
IHcols = ["IH{}".format(i) for i in range(1, 6)] + ["IH{}".format(i) for i in range(7, 10)]
FHcols = ["FH{}".format(i) for i in range(1, 6)]
MHcols = ["MH{}".format(i) for i in range(1, 42)]
MKcols = ["MK{}".format(i) for i in range(1, 49)]
allcols = PIcols + EIcols + IIcols + IHcols + FHcols + MHcols + MKcols

# utility functions

def replaced_columns(columns):
    def replace(name):
        name = re.sub("Product_Info_", "PI", name)
        name = re.sub("Employment_Info_", "EI", name)
        name = re.sub("InsuredInfo_", "II", name)
        name = re.sub("Insurance_History_", "IH", name)
        name = re.sub("Medical_History_", "MH", name)
        name = re.sub("Family_Hist_", "FH", name)
        name = re.sub("Medical_Keyword_", "MK", name)
        return name
    return map(replace, columns)

def replacer_df(series):
    """
        output: DataFrame, index is value, columns are ratio and order
        assume there is no nan
    """
    df = pd.DataFrame(np.array(series), columns=["val"])
    g = df.groupby("val").size().sort_values(ascending=False)
    dfg = pd.DataFrame(g, columns=["size"])
    dfg["ratio"] = dfg["size"] / dfg["size"].sum()
    dfg["order"] = np.arange(len(dfg)).astype(int)
    return dfg

def replace_rarevalues(_data, columns, threshold, repval):
    df = _data.copy()
    for c in columns:
        se = df[c].value_counts()
        se = se[se <= threshold]
        df[c] = np.where(df[c].isin(se.index), repval, df[c])
    return df
