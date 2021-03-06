# -*- coding: utf-8 -*-
"""
Created on Sun Sep 04 14:12:49 2016

@author: admin
"""
from infodens.feature_extractor.feature_extractor import featid, Feature_extractor
from scipy import sparse
import numpy as np
from collections import Counter
import operator
from nltk import ngrams
import subprocess
import os
import codecs
import argparse


class Lang_model_features(Feature_extractor):

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

    def getKenLMScores(self, sentences, langModel):
        __import__('imp').find_module('kenlm')
        import kenlm
        model = kenlm.Model(langModel)
        probab = []
        for sent in sentences:
            if isinstance(sent,list):
                sent = " ".join(sent)
            probab.append([model.score(sent, bos=True, eos=True),
                           model.perplexity(sent)])

        return sparse.lil_matrix(probab)

    def getPynlplScores(self, sentences, langModel):
        import pynlpl.lm.lm as pineApple
        arpaLM = pineApple.ARPALanguageModel(langModel)
        probab = []
        print("Using Pineapple")
        for sent in sentences:
            probab.append([arpaLM.score(sent)])

        return sparse.lil_matrix(probab)

    def getSRILMScores(self, sentsFile, langModel,ngramOrder, srilmBinary, sentCount):
        pplFile = "tempLang{0}{1}.ppl".format(os.path.basename(sentsFile), ngramOrder)
        command = "\"{0}ngram\" -order {1} -lm {2} -ppl {3} -debug 1 -unk> {4}".format(srilmBinary, ngramOrder,
                                                                                       langModel, sentsFile, pplFile)

        subprocess.call(command, shell=True)
        probab = self.extractValues(pplFile, sentCount)
        os.remove(pplFile)
        return sparse.lil_matrix(probab)

    @featid(17)
    def langModelFeat(self, argString, preprocessReq=0):
        '''
        Extracts n-gram Language Model perplexity features.
        '''
        parser = argparse.ArgumentParser(description='LM perplexity args')
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
                langModel = self.preprocessor.buildLanguageModel(ngramOrder)
            return 1

        sentsFile = self.preprocessor.getInputFileName()
        testSentsFile = self.testPreprocessor.getInputFileName()
        srilmBinary, kenlm = self.preprocessor.prep_servs.getBinariesPath()

        if not langModel:
            langModel = self.preprocessor.buildLanguageModel(ngramOrder)

        if srilmBinary and not kenlm:
            probTrain = self.getSRILMScores(sentsFile, langModel, ngramOrder, srilmBinary,
                                            self.preprocessor.getSentCount())
            probTest = self.getSRILMScores(testSentsFile, langModel,ngramOrder, srilmBinary,
                                           self.testPreprocessor.getSentCount())
        else:
            try:
                # Use KenLM
                probTrain = self.getKenLMScores(self.preprocessor.getPlainSentences(), langModel)
                probTest = self.getKenLMScores(self.testPreprocessor.getPlainSentences(), langModel)
            except ImportError:
                # Use pynlpl
                probTrain = self.getPynlplScores(self.preprocessor.gettokenizeSents(), langModel)
                probTest = self.getPynlplScores(self.testPreprocessor.gettokenizeSents(), langModel)

        return probTrain, probTest, "Sentence preplexity"

    def parsePOSArgs(self, args):

        parser = argparse.ArgumentParser(description='POS surprisal args')
        parser.add_argument("-pos_train", help="Path for POS tagged train sentences.",
                            type=str, default="")
        parser.add_argument("-pos_test", help="Path for POS tagged test sentences.",
                            type=str, default="")
        parser.add_argument("-pos_corpus", help="Tagged corpus for language model.",
                            type=str, default="")
        parser.add_argument("-pos_lm", help="Language model of a POS tagged corpus.",
                            type=str, default="")
        parser.add_argument("-ngram", help="Order of language model.",
                            type=int, default=3)

        argsOut = parser.parse_args(args.split())
        return argsOut.pos_train, argsOut.pos_test, argsOut.pos_corpus, argsOut.pos_lm, argsOut.ngram

    @featid(18)
    def langModelPOSFeat(self, argString, preprocessReq=0):
        '''
        Extracts n-gram POS language model perplexity features.
        '''
        # -pos_train -pos_test -pos_corpus -pos_lm -ngram=3
        taggedInput, taggedTest, taggedCorpus, langModel, ngramOrder = self.parsePOSArgs(argString)

        if preprocessReq:
            # Request all preprocessing functions to be prepared
            if not taggedInput:
                taggedInput = self.preprocessor.prep_servs.dumpTokensTofile(
                                            dumpFile="{0}_tagged_Input.txt".format(self.preprocessor.getInputFileName()),
                                                 tokenSents=self.preprocessor.getPOStagged())
            if not taggedTest:
                taggedTest = self.preprocessor.prep_servs.dumpTokensTofile(
                                            dumpFile="{0}_tagged_test.txt".format(self.testPreprocessor.getInputFileName()),
                                                 tokenSents=self.testPreprocessor.getPOStagged())
            if not langModel:
                if not taggedCorpus:
                    taggedCorpus = self.preprocessor.prep_servs.dumpTokensTofile(
                                        dumpFile="{0}_tagged_Corpus.txt".format(self.preprocessor.getCorpusLMName()),
                                                     tokenSents=self.preprocessor.prep_servs.tagPOSfromFile(
                                                         self.preprocessor.getCorpusLMName()
                                                     ))

                # If tagged corpus is empty, just use
                langModel = self.preprocessor.buildLanguageModel(ngramOrder, taggedCorpus, False)

            return 1

        if not taggedInput:
            taggedInput = "{0}_tagged_Input.txt".format(self.preprocessor.getInputFileName())
        if not taggedTest:
            taggedTest = "{0}_tagged_test.txt".format(self.testPreprocessor.getInputFileName())
        if not langModel:
            if not taggedCorpus:
                taggedCorpus = "{0}_tagged_Corpus.txt".format(self.preprocessor.getCorpusLMName())
            langModel = self.preprocessor.buildLanguageModel(ngramOrder, taggedCorpus, False)

        srilmBinary, kenlm = self.preprocessor.prep_servs.getBinariesPath()

        if srilmBinary and not kenlm:
            probTrain = self.getSRILMScores(taggedInput, langModel, ngramOrder, srilmBinary,
                                            self.preprocessor.getSentCount())
            probTest = self.getSRILMScores(taggedTest, langModel, ngramOrder, srilmBinary,
                                           self.testPreprocessor.getSentCount())
        else:
            try:
                # Use KenLM
                probTrain = self.getKenLMScores(self.preprocessor.getPOStagged(taggedInput), langModel)
                probTest = self.getKenLMScores(self.testPreprocessor.getPOStagged(taggedTest), langModel)
            except ImportError:
                # Use pynlpl
                probTrain = self.getPynlplScores(self.preprocessor.getPOStagged(taggedInput), langModel)
                probTest = self.getPynlplScores(self.testPreprocessor.getPOStagged(taggedTest), langModel)

        return probTrain, probTest, "POS Sentence preplexity"
