# -*- coding: utf-8 -*-
"""
Created on Sun Sep 04 14:12:49 2016

@author: admin
"""
from infodens.feature_extractor.feature_extractor import featid, Feature_extractor
from scipy import sparse
import codecs
import argparse
import os
import time
from fastText import train_supervised
from fastText import load_model


class FastText_features(Feature_extractor):

    def roundLabelsRegress(self, regressions):
        ''' This function approximates regression values
        to labels. Enabling the use of fastText with continuous values'''
        labels = []
        for regress in regressions:
            labels.append(round(regress))

        return labels

    def trainFastText(self, sents, labls, argsOut):

        if argsOut.load:
            # load given model
            print("Loading fastText model {0} ..".format(argsOut.load))
            return load_model(argsOut.load)

        outFile = "ft_runTrain{0}.txt".format(time.time())

        labls = self.roundLabelsRegress(labls)

        for i in range(0, len(sents)):
            sents[i] = "__label__{0} {1}".format(labls[i], sents[i])

        with open(outFile, mode="w", encoding="utf-8") as file:
            file.writelines(sents)

        # Get model
        model = train_supervised(input=outFile, epoch=argsOut.epochs, dim=argsOut.dim, bucket=10000000,
                                 lr=argsOut.lr, wordNgrams=argsOut.wordNgrams, verbose=2, minCount=1)
        os.remove(outFile)

        return model

    def getEmbeddings(self, sents, model):
        feats = []
        for sent in sents:
            feats.append(model.get_sentence_vector(sent.strip()))
        return sparse.lil_matrix(feats)

    def parseFastTextArgs(self, args):

        parser = argparse.ArgumentParser(description='Fasttext args')
        parser.add_argument("-epochs", help="Number of training epochs.",
                            type=int, default=10)
        parser.add_argument("-lr", help="learning rate",
                            type=float, default=1.0)
        parser.add_argument("-wordNgrams", help="Ngram features to add",
                            type=int, default=0)
        parser.add_argument("-dim", help="Length of embeddings vector.",
                            type=int, default=100)
        parser.add_argument("-save", help="Name of file to save model to.",
                            type=str, default="")
        parser.add_argument("-load", help="Name of file to load model from.",
                            type=str, default="")

        argsOut = parser.parse_args(args.split())
        return argsOut

    @featid(801)
    def learned_embeddings(self, argString, preprocessReq=0):
        '''
        Extracts learned representation from fastText
        '''

        if preprocessReq:
            self.preprocessor.getPlainSentences()
            self.testPreprocessor.getPlainSentences()
            return 1

        args = self.parseFastTextArgs(argString)

        classes = self.preprocessor.prep_servs.preprocessClassID(
            self.preprocessor.getInputClassesFile())

        # Train on train sents only
        sents = self.preprocessor.getPlainSentences()

        model = self.trainFastText(sents, classes, args)

        testSents = self.testPreprocessor.getPlainSentences()

        trainFeats = self.getEmbeddings(sents, model)
        testFeats = self.getEmbeddings(testSents, model)

        if args.save:
            model.save_model(args.save)

        print("Learned embeddings length : {0}".format((trainFeats.get_shape()[1])))

        return trainFeats, testFeats, "fastText embeddings"

