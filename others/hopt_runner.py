# -*- coding: utf-8 -*-

import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import model_util
from util import Writer

class HyperOptRunner:

    def __init__(self):
        self.fpath = None
        self.space = None
        self.max_evals = None
        self.output_items = None
        self.init2()
        self.columns = sorted(self.space.keys())

    def init2(self):
        pass

    def calculate_loss(self, params):
        """
            input: prms, 
            output: dict - loss and output_items
        """
        pass

    def run(self):
        trials = Trials()
        Writer.create_empty(self.fpath)
        Writer.append_line_list(self.fpath, [c for c in self.columns] + self.output_items)

        best = fmin(self.score, self.space, algo=tpe.suggest, trials=trials, max_evals=self.max_evals)
        print best

    def score(self, params):
        outputs = self.calculate_loss(params)
        Writer.append_line_list(self.fpath, [params[c] for c in self.columns] + [outputs[c] for c in self.output_items])
        return {'loss': outputs["loss"], 'status': STATUS_OK}
