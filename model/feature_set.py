import sys
sys.path += [".", "utils"]
from feat_transformer import PercentileBin, MeanAbs, OneHot
from feature_set_enslist import enslist

# feature definitions
feature_defs = {}

feature_defs["f1"] = [("A", ["basic"], [])]

feature_defs["fnn1"] = [
        ("A", ["PI4"], []),
        ("B", ["Ins_Age"], []),
        ("C", ["build"], []),
        ("D", ["EIcont"], []),
        ("E", ["FHcont"], []),
        ("F", ["IHcont"], []),
        ("G", ["MHdcr"], []),
        ("H", ["MKdummy"], []),
        ("I", ["cat1PIb"], []),
        ("J", ["cat1EIb"], []),
        ("K", ["cat1IIb"], []),
        ("L", ["cat1IHb"], []),
        ("M", ["cat1FHb"], []),
        ("N", ["cat1MHb"], []),
        ("O", ["PI2b", "PI3b"], []),
        ("P", ["EI2b"], []),
        ("Q", ["II3b"], []),
        ("R", ["MH2b"], []),
        ("S", ["isnan"], []),
        ("T", ["MKsum"], []),
        ("U", ["PI2c", "PI2n"], []),
    ]

feature_defs["en1"] = [("Z",  enslist["en1"], [])]

feature_defs["f1_en1"] = list(feature_defs["f1"])
feature_defs["f1_en1"] += [("Z",  enslist["en1"], [])]

feature_defs["fvw1"] = list(feature_defs["fnn1"])

feature_defs["fvw2"] = list(feature_defs["fnn1"])
feature_defs["fvw2"] += [
        ("B", ["Ins_Age"], [MeanAbs()]),
        ("C", ["build"], [MeanAbs()]),
        ("D", ["EIcont"], [MeanAbs()]),
        ("E", ["FHcont"], [MeanAbs()]),
        ("F", ["IHcont"], [MeanAbs()]),
        ("G", ["MHdcr"], [MeanAbs()]),
        ("B", ["Ins_Age"], [PercentileBin()]),
        ("C", ["build"], [PercentileBin()]),
        ("D", ["EIcont"], [PercentileBin()]),
        ("E", ["FHcont"], [PercentileBin()]),
        ("F", ["IHcont"], [PercentileBin()]),
        ("G", ["MHdcr"], [PercentileBin()]),
]

# vowpal wabbit interaction list
vw_inter_list = {}
vw_inter_list["fvw1"] = ""
vw_inter_list["fvw2"] = ""