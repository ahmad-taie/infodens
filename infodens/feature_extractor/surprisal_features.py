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
import math
import subprocess
import os
import codecs
import argparse


class Surprisal_features(Feature_extractor):

    def extractValues(self, srilmOutput, sentCount):
        feats = []
        with codecs.open(srilmOutput, encoding='utf-8') as sents:
            for line in sents:
                if "logprob=" in line:
                    line = str(line).strip().split(" ")
                    if len(feats) < sentCount:
                        tmp = 0.0
                        if line[3] != "undefined":
                            log10Prob = -float(line[3])
                            log2prob = log10Prob/math.log(2,10)
                            tmp = log2prob
                        else:
                            tmp = np.float32(0.0)
                        feats.append(tmp)
        return feats

    def perplexity(self, sentence, sentScore):
        """
        Compute perplexity of a sentence.
        @param sentence One full sentence to score.  Do not include <s> or </s>.
        (Reused from kenlm pyx)

        """
        def as_str(data):
            return data.encode('utf8')

        words = len(as_str(sentence).split()) + 1  # For </s>

        return 2.0 ** (-sentScore / words)

    def getKenLMScores(self, sentences, langModel):
        __import__('imp').find_module('kenlm')
        import kenlm
        model = kenlm.Model(langModel)
        probab = []
        for sent in sentences:
            if isinstance(sent,list):
                sent = " ".join(sent)
            log10Prob = model.score(sent, bos=True, eos=True)
            log2prob = log10Prob / math.log(2, 10)
            probab.append([log2prob, self.perplexity(sent, log2prob)])

        return sparse.lil_matrix(probab)

    def getPynlplScores(self, sentences, langModel):
        import pynlpl.lm.lm as pineApple
        arpaLM = pineApple.ARPALanguageModel(langModel)
        probab = []
        print("Using Pineapple")
        for sent in sentences:
            log10Prob = arpaLM.score(sent)
            log2prob = log10Prob / math.log(2, 10)
            probab.append([log2prob])

        return sparse.lil_matrix(probab)

    def getSRILMScores(self, sentsFile, langModel,ngramOrder, srilmBinary, sentCount):
        pplFile = "tempLang{0}{1}.ppl".format(os.path.basename(sentsFile), ngramOrder)
        command = "\"{0}ngram\" -order {1} -lm {2} -ppl {3} -debug 1 -unk> {4}".format(srilmBinary, ngramOrder,
                                                                                       langModel, sentsFile, pplFile)

        subprocess.call(command, shell=True)
        probab = self.extractValues(pplFile, sentCount)
        os.remove(pplFile)
        return sparse.lil_matrix(probab)

    @featid(20)
    def surplangModelFeat(self, argString, preprocessReq=0):
        '''
        Extracts n-gram Language Model preplexity features.
        '''
        parser = argparse.ArgumentParser(description='Language Model preplexity  args')
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

        return probTrain, probTest, "Sentence surprisal"

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

    @featid(21)
    def surplangModelPOSFeat(self, argString, preprocessReq=0):
        '''
        Extracts n-gram POS language model preplexity features.
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

        return probTrain, probTest, "POS Sentence surprisal"

    def getSplits(self, counts, sumcounts, splitCount):
        splits = []
        tmpsum = 0
        prevsum = 0

        for i in range(0, len(counts)):
            tmpsum += counts[i]
            if tmpsum > sumcounts:
                if np.abs(sumcounts - prevsum) <= np.abs(sumcounts - tmpsum):
                    tmpsum = counts[i]
                    splits.append(i)
                else:
                    tmpsum = 0
                    splits.append(i + 1)
                if len(splits) == splitCount-1:
                    return splits
            prevsum = tmpsum
        return splits

    def ngramArgCheck(self, argString):

        parser = argparse.ArgumentParser(description='Ngram quantile surprisal args')
        parser.add_argument("-ngram", help="Order of ngrams.",
                            type=int, default=1)
        parser.add_argument("-cutoff", help="cuttoff frequency",
                            type=int, default=1)
        parser.add_argument("-n_quantiles", help="number of quantiles",
                            type=int, default=4)
        argsOut = parser.parse_args(argString.split())

        return argsOut.ngram, argsOut.cutoff, argsOut.n_quantiles

    def getQuantiles(self, listOfSentences, n, quantile, finNgram):

        ngramFeatures = sparse.lil_matrix((len(listOfSentences), quantile + 1))

        for i in range(len(listOfSentences)):
            ngramsVocab = Counter(ngrams(listOfSentences[i], n))
            lenSent = 0
            for ngramEntry in ngramsVocab:
                ## Keys
                ngramIndex = finNgram.get(ngramEntry, -1)
                if ngramIndex >= 0:
                    ngramIndex -= 1
                    toAdd = ngramsVocab[ngramEntry]
                    ngramFeatures[i, ngramIndex] += toAdd
                    lenSent += toAdd
                else:
                    # OOV word (cut-off
                    toAdd = ngramsVocab[ngramEntry]
                    ngramFeatures[i, -1] += toAdd
                    lenSent += toAdd

            if lenSent:
                for j in range(0, quantile+1):
                    ngramFeatures[i, j] /= lenSent

        return ngramFeatures

    @featid(19)
    def quantileNgramSurprisal(self, argString, preprocessReq=0):

        # Handle preprocessing requests and exit
        if preprocessReq:
            self.preprocessor.gettokenizeSents()
            self.testPreprocessor.gettokenizeSents()
            return 1

        n, freq, nQuantas = self.ngramArgCheck(argString)

        tokensCorpus = self.preprocessor.gettokenizeSents()

        finNgram, numberOfFeatures = self.preprocessor.prep_servs.buildNgrams(n, freq,
                                                                              tokensCorpus, indexing=False)

        print("Ngrams built.")

        if numberOfFeatures == 0:
            print("Cut-off too high, no ngrams passed it.")
            return []

        listNgrams = sorted(finNgram.items(), key=operator.itemgetter(1), reverse=True)
        ngramsKeys, ngramCounts = zip(*listNgrams)

        ngramCounts = list(ngramCounts)
        splitSum = int(sum(ngramCounts)/nQuantas)

        ngramsKeys = list(ngramsKeys)

        indecesSplit = self.getSplits(ngramCounts, splitSum, nQuantas)
        #print(indecesSplit)

        splitIndex = 0
        quantile = 1
        finNgram = {}

        for i in range(0, len(ngramsKeys)):
            if splitIndex < len(indecesSplit) and i >= indecesSplit[splitIndex]:
                quantile += 1
                splitIndex += 1
            finNgram[ngramsKeys[i]] = quantile

        listOfSentences = self.preprocessor.gettokenizeSents()
        testSentences = self.testPreprocessor.gettokenizeSents()
        # +1 for OOV

        print("Extracting ngram feats.")
        trainFeatures = self.getQuantiles(listOfSentences, n, quantile, finNgram)
        testFeatures = self.getQuantiles(testSentences, n, quantile, finNgram)
        print("Finished ngram features.")

        return trainFeatures, testFeatures, "Ngram quantile"
