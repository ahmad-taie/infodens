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

    @featid(17)
    def langModelFeat(self, argString, preprocessReq=0):
        '''
        Extracts n-gram Language Model preplexity features.
        '''
        ngramOrder = 3
        langModel = 0
        # Binary1/0,ngramOrder,LMFilePath(ifBinary1)
        arguments = argString.split(',')
        if(int(arguments[0])):
            # Use given langModel
            langModel = "\"{0}\"".format(arguments[-1])

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

        if srilmBinary and not kenlm:
            pplFile = "tempLang{0}{1}.ppl".format(os.path.basename(sentsFile), ngramOrder)
            command = "\"{0}ngram\" -order {1} -lm {2} -ppl {3} -debug 1 -unk> {4}".format(srilmBinary, ngramOrder,
                                                                                         langModel, sentsFile, pplFile)

            subprocess.call(command, shell=True)
            probab = self.extractValues(pplFile, self.preprocessor.getSentCount())
            os.remove(pplFile)
            return sparse.lil_matrix(probab)
        else:
            try:
                __import__('imp').find_module('kenlm')
                import kenlm
                model = kenlm.Model(langModel)
                probab = []
                for sent in self.preprocessor.getPlainSentences():
                    probab.append([model.score(sent, bos=True, eos=True),
                                   model.perplexity(sent)])
                output = sparse.lil_matrix(probab)
                return output
            except ImportError:
                import pynlpl.lm.lm as pineApple
                arpaLM = pineApple.ARPALanguageModel(langModel)
                probab = []
                for sent in self.preprocessor.gettokenizeSents():
                    probab.append([arpaLM.score(sent)])
                output = sparse.lil_matrix(probab)
                return output

    @featid(18)
    def langModelPOSFeat(self, argString, preprocessReq=0):
        '''
        Extracts n-gram POS language model preplexity features.
        '''
        ngramOrder = 3
        langModel = ""
        taggedInput = ""
        taggedCorpus = ""
        # TaggedInput1/0,LM0/1,taggedCorpus0/1,ngramOrder(,TaggedPOSfile(ifTaggedInp1),
        # LMFilePath(ifLM1),taggedCorpus(if LM0&TaggedCorpus1))
        arguments = argString.split(',')
        if int(arguments[0]):
            # Use file of tagged sents (last argument)
            taggedInput = "\"{0}\"".format(arguments[4])
        if int(arguments[1]):
            # Next argument
            langModel = "\"{0}\"".format(arguments[4+int(arguments[0])])
        elif int(arguments[2]):
            taggedCorpus = "\"{0}\"".format(arguments[4+int(arguments[0])])

        ngramOrder = int(arguments[3])

        if preprocessReq:
            # Request all preprocessing functions to be prepared
            if not taggedInput:
                taggedInput = self.prep_servs.dumpTokensTofile(
                                            dumpFile="{0}_tagged_Input.txt".format(self.preprocessor.getInputFileName()),
                                                 tokenSents=self.preprocessor.getPOStagged())
            if not langModel:
                if not taggedCorpus:
                    taggedCorpus = self.prep_servs.dumpTokensTofile(
                                        dumpFile="{0}_tagged_Corpus.txt".format(self.preprocessor.getCorpusLMName()),
                                                     tokenSents=self.prep_servs.tagPOSfromFile(
                                                         self.preprocessor.getCorpusLMName()
                                                     ))

                # If tagged corpus is empty, just use
                langModel = self.preprocessor.buildLanguageModel(ngramOrder, taggedCorpus, False)

            return 1

        if not taggedInput:
            taggedInput = "{0}_tagged_Input.txt".format(self.preprocessor.getInputFileName())
        if not langModel:
            if not taggedCorpus:
                taggedCorpus = "{0}_tagged_Corpus.txt".format(self.preprocessor.getCorpusLMName())
            langModel = self.preprocessor.buildLanguageModel(ngramOrder, taggedCorpus, False)

        srilmBinary, kenlm = self.preprocessor.getBinariesPath()

        if srilmBinary and not kenlm:
            pplFile = "tempLang{0}{1}.ppl".format(os.path.basename(taggedInput), ngramOrder)

            command = "\"{0}ngram\" -order {1} -lm {2} -ppl {3} -debug 1 -unk> {4}".format(srilmBinary, ngramOrder,
                                                                                           langModel, taggedInput, pplFile)

            subprocess.call(command, shell=True)
            probab = self.extractValues(pplFile, self.preprocessor.getSentCount())
            os.remove(pplFile)
            return sparse.lil_matrix(probab)
        else:
            try:
                __import__('imp').find_module('kenlm')
                import kenlm
                model = kenlm.Model(langModel)
                probab = []
                for sent in self.preprocessor.getPOStagged():
                    probab.append([model.score(sent, bos=True, eos=True),
                                   model.perplexity(sent)])
                output = sparse.lil_matrix(probab)
                return output
            except ImportError:
                import pynlpl.lm.lm as pineApple
                arpaLM = pineApple.ARPALanguageModel(langModel)
                probab = []
                for sent in self.preprocessor.getPOStagged():
                    probab.append([arpaLM.score(sent)])
                output = sparse.lil_matrix(probab)
                return output

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

        #format : 1,2,5

        status = 1
        n = 0
        freq = 1
        #default of 4 splits
        splits = 4

        argStringList = argString.split(',')
        if argStringList[0].isdigit():
            n = int(argStringList[0])
        else:
            print('Error: n should be an integer')
            status = 0
        if len(argStringList) > 1:
            if argStringList[1].isdigit():
                freq = int(argStringList[1])
            else:
                print('Error: frequency should be an integer')
                status = 0
            # splits
            if len(argStringList) > 2:
                if argStringList[2].isdigit():
                    splits = int(argStringList[2])
                else:
                    print('Error: splits should be an integer')
                    status = 0

        return status, n, freq, splits

    @featid(19)
    def quantileNgramSurprisal(self, argString, preprocessReq=0):

        # Handle preprocessing requests and exit
        if preprocessReq:
            self.preprocessor.gettokenizeSents()
            return 1

        status, n, freq, nQuantas = self.ngramArgCheck(argString)
        if not status:
            # Error in argument.
            return

        tokensCorpus = self.preprocessor.prep_servs.getFileTokens(self.preprocessor.getCorpusLMName())

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
        ngramFeatures = sparse.lil_matrix((len(listOfSentences), quantile))

        print("Extracting ngram feats.")

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

            if lenSent:
                for j in range(0, quantile):
                    ngramFeatures[i, j] /= lenSent

        print("Finished ngram features.")

        return ngramFeatures
