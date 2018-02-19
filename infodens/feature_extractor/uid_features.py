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


class UID_features(Feature_extractor):

    def extractValues(self, srilmOutput, sentCount):
        feats = []
        with codecs.open(srilmOutput, encoding='utf-8') as sents:
            for line in sents:
                if "logprob=" in line:
                    line = str(line).strip().split(" ")
                    if len(feats) < sentCount:
                        tmp = []
                        for i in [3, 5, 7]:
                            if line[i] != "undefined":
                                tmp.append(np.float32(line[i]))
                            else:
                                tmp.append(np.float32(0.0))
                        feats.append(tmp)
        return feats

    @featid(77)
    def uid_variance(self, argString, preprocessReq=0):
        '''
        Extracts n-gram Language Model surprisal variance.
        '''
        ngramOrder = 3
        langModel = 0
        # Binary1/0,ngramOrder,LMFilePath(ifBinary1)
        arguments = argString.split(',')
        if(int(arguments[0])):
            # Use given langModel
            # (problems from spaces in path can be avoided by quotes)
            langModel = "{0}".format(arguments[-1])

        ngramOrder = int(arguments[1])

        if preprocessReq:
            # Request all preprocessing functions to be prepared
            if not langModel:
                langModel = self.preprocessor.buildLanguageModel(ngramOrder)
            self.preprocessor.getInputFileName()
            self.preprocessor.getBinariesPath()
            return 1

        sentsFile = self.preprocessor.getInputFileName()
        srilmBinary, kenlm = self.preprocessor.getBinariesPath()

        if not langModel:
            langModel = self.preprocessor.buildLanguageModel(ngramOrder)

        __import__('imp').find_module('kenlm')
        import kenlm
        model = kenlm.Model(langModel)
        vars = []
        for toks in self.preprocessor.gettokenizeSents():
            scores = []
            for i in range(1, len(toks) + 1):
                scores.append(-model.score(" ".join(toks[:i]), eos=False))
            scores.append(-model.score(" ".join(toks), eos=True))
            vars.append(np.var(scores))

        output = sparse.lil_matrix(vars).transpose()
        return output, "UID surprisal variance."

