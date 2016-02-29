from sklearn.cross_validation import KFold, StratifiedKFold
import os
import pickle
from util import Files

class Config:

    # basic attributes
    def __init__(self, run_type, version="temp", test=False):
        self.folds_num = 5
        self.version = version
        self.run_type = run_type
        self.test = test

        # try in single fold
        if self.run_type == "explore":
            self.folds = [("A", 0)]

        # run in cv folds and all data
        if self.run_type == "production":
            self.folds = [("A", i) for i in range(self.folds_num)] + [None]

        # run in another fold
        if self.run_type == "another":
            # self.folds = [("B", i) for i in range(self.folds_num)]
            self.folds = [("B", i) for i in range(1)]

        # classes should be 1-8 in this problem
        self.classes = 8

    # folds for cross validation
    def create_cvfold(self, labels, char):
        # kf = KFold(len(labels), self.folds, random_state=71, shuffle=True)
        seeds = {"A": 71, "B": 72}
        kf = StratifiedKFold(labels, self.folds_num, random_state=seeds[char], shuffle=True)
        kf = list(kf)
        return kf

    # get path name corresponded to i_fold
    def get_feat_folder(self, i_fold):
        return os.path.join("../model", "feat", self.get_folder_str(i_fold))

    def get_feat_path(self, fname, i_fold):
        return os.path.join(self.get_feat_folder(i_fold), fname)

    def get_model_folder(self, model_name, i_fold):
        return os.path.join("../model", "model", model_name, self.get_folder_str(i_fold))

    def get_model_path(self, model_name, fname, i_fold):
        return os.path.join(self.get_model_folder(model_name, i_fold), fname)

    def get_folder_str(self, i_fold):
        if i_fold is None: return "all"
        if i_fold == "common": return "common"
        return "fold{}{}".format(i_fold[0], i_fold[1])
        
    # save config file
    def dump(self, fpath="../model/config.pkl"):
        with open(fpath, "wb") as f:
            pickle.dump(self, f)

    # load config file
    @classmethod
    def load(cls, fname="../model/config.pkl"):
        with open(fname, "rb") as f:
            obj = pickle.load(f)
        return obj
