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
from fastText import train_supervised


def trainFastText(sents, labls):

    outFile = "ft_runTrain.txt"

    for i in range(0, len(sents)):
        sents[i] = "__label__{0} {1}".format(labls[i], sents[i])

    with open(outFile, "w") as file:
        file.writelines(sents)

    # Get model
    model = train_supervised(input=outFile, epoch=25, lr=1.0, wordNgrams=2,
                                      verbose=2, minCount=1)
    os.remove(outFile)

    return model


class FastText_features(Feature_extractor):

    @featid(801)
    def learned_embeddings(self, argString, preprocessReq=0):
        '''
        Extracts learned representation from fastText
        '''

        if preprocessReq:
            self.preprocessor.getPlainSentences()
            return 1

        feats = []
        classes = self.preprocessor.prep_servs.preprocessClassID(
            self.preprocessor.getInputClassesFile())

        sents = self.preprocessor.getPlainSentences()

        model = trainFastText(sents, classes)

        for sent in sents:
            feats.append(model.get_sentence_vector(sent.strip()))

        output = sparse.lil_matrix(feats)

        print("Learned embeddings length : {0}".format(len(feats[0])))

        return output, "fastText embeddings"

