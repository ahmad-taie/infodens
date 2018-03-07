# -*- coding: utf-8 -*-
"""
Created on Sun Sep 04 14:12:49 2016

@author: admin
"""
from infodens.feature_extractor.feature_extractor import featid, Feature_extractor
from scipy import sparse
import numpy as np
import subprocess
import os
import codecs
import argparse


class UID_features(Feature_extractor):

    def extractValues(self, trainSents, model):
        vars = []
        for toks in trainSents:
            scores = []
            for i in range(1, len(toks) + 1):
                scores.append(-model.score(" ".join(toks[:i]), eos=False))
            scores.append(-model.score(" ".join(toks), eos=True))
            vars.append(np.var(scores))
        return sparse.lil_matrix(vars).transpose()

    @featid(77)
    def uid_variance(self, argString, preprocessReq=0):
        '''
        Extracts n-gram Language Model surprisal variance.
        '''
        parser = argparse.ArgumentParser(description='Variance in sentence probability args')
        parser.add_argument("-lm", help="Language model of a corpus.",
                            type=str, default="")
        parser.add_argument("-ngram", help="Order of language model.",
                            type=int, default=3)

        argsOut = parser.parse_args(argString.split())
        ngramOrder = argsOut.ngram
        langModel = argsOut.lm

        if preprocessReq:
            # Request all preprocessing functions to be prepared
            if not langModel:
                self.preprocessor.buildLanguageModel(ngramOrder)
                self.preprocessor.gettokenizeSents()
                self.testPreprocessor.gettokenizeSents()
            return 1

        if not langModel:
            langModel = self.preprocessor.buildLanguageModel(ngramOrder)

        __import__('imp').find_module('kenlm')
        import kenlm
        model = kenlm.Model(langModel)
        trainUID = self.extractValues(self.preprocessor.gettokenizeSents(), model)
        testUID = self.extractValues(self.testPreprocessor.gettokenizeSents(), model)

        return trainUID, testUID, "UID surprisal variance."

